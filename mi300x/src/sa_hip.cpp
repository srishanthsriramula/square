// High-Throughput Geometric Simulated Annealing Engine for AMD Instinct MI300X (ROCm / HIP)
// Optimized for CDNA3 architecture (304 Compute Units, 19,456 stream cores, Wavefront 64).
// Ported and enhanced from Francisco Couzo & Ellsworth's cooperative GPU annealing formulation.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define WARP_SIZE 64
#define FULL_MASK 0xffffffffffffffffULL
#define SYNC_WARP(m) __builtin_amdgcn_wave_barrier()
#else
#include <cuda_runtime.h>
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipSetDeviceFlags cudaSetDeviceFlags
#define hipDeviceScheduleBlockingSync cudaDeviceScheduleBlockingSync
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMemset cudaMemset
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#define SYNC_WARP(m) __syncwarp(m)
#endif

#ifndef LPP
#define LPP 32          // 32 lanes per packing
#endif
#ifndef PPB
#define PPB 2           // 2 packings per block -> 64 threads (1 CDNA3 wavefront!)
#endif
#define BLOCK (LPP * PPB)

#if LPP == WARP_SIZE
#define GMASK(g) FULL_MASK
#else
#define GMASK(g) ((((1ULL << LPP) - 1ULL)) << ((threadIdx.x & (WARP_SIZE - 1)) - (g)))
#endif

// PCG32 Random Number Generator per-chain
struct Rng { unsigned long long s, inc; };

__device__ __forceinline__ unsigned rnd(Rng &r) {
    unsigned long long old = r.s;
    r.s = old * 6364136223846793005ULL + r.inc;
    unsigned xs = (unsigned)(((old >> 18) ^ old) >> 27);
    unsigned rot = (unsigned)(old >> 59);
    return (xs >> rot) | (xs << ((32 - rot) & 31));
}

__device__ __forceinline__ float rndf(Rng &r) {
    return (rnd(r) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float rndn(Rng &r) {
    return (rndf(r) + rndf(r) + rndf(r) - 1.5f) * 1.1547f;
}

__device__ __forceinline__ unsigned long long splitmix(unsigned long long &x) {
    x += 0x9e3779b97f4a7c15ULL;
    unsigned long long z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ int pick(Rng &r, int n) {
    return (int)(((unsigned long long)rnd(r) * (unsigned long long)n) >> 32);
}

// Penetration depth of two unit squares via Separating Axis Theorem projections
__device__ __forceinline__ float depth(float dx, float dy,
                                       float ca, float sa, float cb, float sb) {
    if (fmaf(dx, dx, dy * dy) >= 2.0f) return 0.0f;
    float cd = fmaf(ca, cb, sa * sb);
    float sd = fmaf(ca, sb, -sa * cb);
    float H = 0.5f + 0.5f * (fabsf(cd) + fabsf(sd));
    float p = fabsf(fmaf(dx, ca, dy * sa));
    p = fmaxf(p, fabsf(fmaf(dx, -sa, dy * ca)));
    p = fmaxf(p, fabsf(fmaf(dx, cb, dy * sb)));
    p = fmaxf(p, fabsf(fmaf(dx, -sb, dy * cb)));
    float d = H - p;
    return d > 0.0f ? d : 0.0f;
}

__device__ __forceinline__ float wall(float x, float y, float c, float s, float side) {
    float h = 0.5f * (fabsf(c) + fabsf(s));
    float e = 0.0f, v;
    v = h - x;        if (v > 0.0f) e = fmaf(v, v, e);
    v = x + h - side; if (v > 0.0f) e = fmaf(v, v, e);
    v = h - y;        if (v > 0.0f) e = fmaf(v, v, e);
    v = y + h - side; if (v > 0.0f) e = fmaf(v, v, e);
    return e;
}

__device__ __forceinline__ float group_sum(float e, int g) {
#if LPP > 1
    #pragma unroll
    for (int o = LPP / 2; o; o >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        e += __shfl_xor(e, o, LPP);
#else
        unsigned m = GMASK(g);
        e += __shfl_xor_sync(m, e, o, LPP);
#endif
    }
#endif
    return e;
}

__device__ __forceinline__ float viol_of(int i, float xi, float yi, float ci, float si,
                                         const float4 *q, int n, float side, int g) {
    float e = 0.0f;
    for (int j = g; j < n; j += LPP) {
        if (j == i) continue;
        float4 b = q[j];
        float d = depth(b.x - xi, b.y - yi, ci, si, b.z, b.w);
        e = fmaf(d, d, e);
    }
    e = group_sum(e, g);
    return e + wall(xi, yi, ci, si, side);
}

__device__ __forceinline__ float wall_total(const float4 *q, int n, float side, int g) {
    float e = 0.0f;
    for (int i = g; i < n; i += LPP) {
        float4 a = q[i];
        e += wall(a.x, a.y, a.z, a.w, side);
    }
    return group_sum(e, g);
}

__device__ float pair_total(const float4 *q, int n, int g) {
    float e = 0.0f;
    for (int i = 0; i < n; i++) {
        float4 a = q[i];
        for (int j = i + 1 + g; j < n; j += LPP) {
            float4 b = q[j];
            float d = depth(b.x - a.x, b.y - a.y, a.z, a.w, b.z, b.w);
            e = fmaf(d, d, e);
        }
    }
    return group_sum(e, g);
}

__device__ __forceinline__ void depth_grad(float dx, float dy,
                                           float ca, float sa, float cb, float sb,
                                           float *gx, float *gy, float *gt, float *e) {
    if (fmaf(dx, dx, dy * dy) >= 2.0f) return;
    float cd = fmaf(ca, cb, sa * sb);
    float sd = fmaf(ca, sb, -sa * cb);
    float H = 0.5f + 0.5f * (fabsf(cd) + fabsf(sd));
    float ux[4] = { ca, -sa, cb, -sb };
    float uy[4] = { sa,  ca, sb,  cb };
    int best = 0;
    float p = -1.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float v = fabsf(fmaf(dx, ux[k], dy * uy[k]));
        if (v > p) { p = v; best = k; }
    }
    float d = H - p;
    if (d <= 0.0f) return;
    *e += d * d;
    float sig = (fmaf(dx, ux[best], dy * uy[best]) >= 0.0f) ? 1.0f : -1.0f;
    float w = 2.0f * d;
    *gx += w * sig * ux[best];
    *gy += w * sig * uy[best];
    float sgc = cd >= 0.0f ? 1.0f : -1.0f;
    float sgs = sd >= 0.0f ? 1.0f : -1.0f;
    float dH = 0.5f * (sgc * sd - sgs * cd);
    float dp = 0.0f;
    if (best == 0)      dp = sig * fmaf(dx, -sa, dy * ca);
    else if (best == 1) dp = sig * fmaf(dx, -ca, -dy * sa);
    *gt += w * (dH - dp);
}

__device__ float quench(float4 *q, float *th, float &side, float *gx, float *gy,
                        float *gt, int n, float mu, int iters, int g) {
    float step = 1e-5f;
    float f = 1e30f;
    for (int it = 0; it < iters; it++) {
        float e = 0.0f, gs = 0.0f;
        for (int i = g; i < n; i += LPP) {
            float4 a = q[i];
            float ex = 0, ey = 0, et = 0, ei = 0;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                float4 b = q[j];
                depth_grad(b.x - a.x, b.y - a.y, a.z, a.w, b.z, b.w, &ex, &ey, &et, &ei);
            }
            e += 0.5f * ei;
            float h = 0.5f * (fabsf(a.z) + fabsf(a.w));
            float dh = 0.5f * ((a.z >= 0 ? -a.w : a.w) + (a.w >= 0 ? a.z : -a.z));
            float v;
            v = h - a.x;        if (v > 0) { e += v * v; ex -= 2 * v; et += 2 * v * dh; }
            v = a.x + h - side; if (v > 0) { e += v * v; ex += 2 * v; et += 2 * v * dh; gs -= 2 * v; }
            v = h - a.y;        if (v > 0) { e += v * v; ey -= 2 * v; et += 2 * v * dh; }
            v = a.y + h - side; if (v > 0) { e += v * v; ey += 2 * v; et += 2 * v * dh; gs -= 2 * v; }
            gx[i] = ex; gy[i] = ey; gt[i] = et;
        }
        e = group_sum(e, g);
        gs = group_sum(gs, g);
        float f0 = side + mu * e;
        float ds = 1.0f + mu * gs;
        SYNC_WARP(GMASK(g));

        bool moved = false;
        for (int bt = 0; bt < 12; bt++) {
            for (int i = g; i < n; i += LPP) {
                float t = th[i] - step * mu * gt[i];
                th[i] = t;
                float4 a = q[i];
                a.x -= step * mu * gx[i];
                a.y -= step * mu * gy[i];
                __sincosf(t, &a.w, &a.z);
                q[i] = a;
            }
            float ns = side - step * ds;
            SYNC_WARP(GMASK(g));
            float f1 = ns + mu * (pair_total(q, n, g) + wall_total(q, n, ns, g));
            if (f1 <= f0) { side = ns; f = f1; moved = true; step *= 1.3f; break; }
            for (int i = g; i < n; i += LPP) {
                float t = th[i] + step * mu * gt[i];
                th[i] = t;
                float4 a = q[i];
                a.x += step * mu * gx[i];
                a.y += step * mu * gy[i];
                __sincosf(t, &a.w, &a.z);
                q[i] = a;
            }
            SYNC_WARP(GMASK(g));
            step *= 0.35f;
        }
        if (!moved) break;
    }
    return f;
}

__global__ __launch_bounds__(BLOCK) void anneal(
        const float *seedcfg, float *gx2, float *gy2, float *gt2,
        float *gbest, int n, int sweeps, float side0,
        float perturb, float T0, float mu0, float mu1,
        unsigned seed, unsigned long long *gmoves, int msrc, int *gkept,
        int qstages, float cool, float mstep, float tele) {
    extern __shared__ char shm[];
    int tid = threadIdx.x;
    int g = tid % LPP;
    int slot = tid / LPP;
    int inst = blockIdx.x * PPB + slot;

    float4 *q = (float4 *)shm + (size_t)slot * n;
    float *th = (float *)((float4 *)shm + (size_t)PPB * n) + (size_t)slot * 4 * n;
    float *gx = th + n, *gy = gx + n, *gt = gy + n;

    unsigned long long sm = ((unsigned long long)seed << 32) ^ (unsigned long long)inst;
    unsigned long long smk = sm;
    Rng st;
    st.inc = (((unsigned long long)inst << 33) | ((unsigned long long)g << 1)) | 1ULL;
    st.s = splitmix(sm);
    rnd(st);

    float side = side0;
    if (seedcfg && msrc > n) {
        int *idx = (int *)((char *)shm + (size_t)PPB * n * (sizeof(float4) + 4 * sizeof(float)))
                 + (size_t)slot * msrc;
        if (g == 0) {
            for (int i = 0; i < msrc; i++) idx[i] = i;
            for (int i = 0; i < n; i++) {
                int j = i + pick(st, msrc - i);
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
        SYNC_WARP(GMASK(g));
        for (int i = g; i < n; i += LPP) {
            int k = idx[i];
            float t = seedcfg[3 * k + 2] + rndn(st) * perturb;
            float4 a;
            a.x = seedcfg[3 * k] + rndn(st) * perturb;
            a.y = seedcfg[3 * k + 1] + rndn(st) * perturb;
            __sincosf(t, &a.w, &a.z);
            q[i] = a; th[i] = t;
            if (gkept) gkept[(size_t)inst * n + i] = k;
        }
    } else if (seedcfg) {
        for (int i = g; i < n; i += LPP) {
            float t = seedcfg[3 * i + 2] + rndn(st) * perturb;
            float4 a;
            a.x = seedcfg[3 * i] + rndn(st) * perturb;
            a.y = seedcfg[3 * i + 1] + rndn(st) * perturb;
            __sincosf(t, &a.w, &a.z);
            q[i] = a; th[i] = t;
        }
    } else {
        for (int i = g; i < n; i += LPP) {
            float t = rndf(st) * 1.5707963f;
            float4 a;
            a.x = 0.5f + rndf(st) * (side - 1.0f);
            a.y = 0.5f + rndf(st) * (side - 1.0f);
            __sincosf(t, &a.w, &a.z);
            q[i] = a; th[i] = t;
        }
    }
    SYNC_WARP(GMASK(g));

    st.inc = (((unsigned long long)inst << 1) | 1ULL);
    st.s = splitmix(smk) ^ 0x2545f4914f6cdd1dULL;
    rnd(st);

    unsigned long long moves = 0;
    for (int sweep = 0; sweep < sweeps; sweep++) {
        float frac = (float)sweep / (float)sweeps;
        float T = T0 * __expf(-cool * frac);
        float step = mstep * __expf(-5.0f * frac);
        float mu = mu0 * __expf(frac * __logf(mu1 / mu0));

        for (int m = 0; m < n; m++) {
            int i = pick(st, n);
            float4 a = q[i];
            float e0 = viol_of(i, a.x, a.y, a.z, a.w, q, n, side, g);

            float nx, ny, nt;
            if (rndf(st) < tele) {
                nx = 0.5f + rndf(st) * (side - 1.0f);
                ny = 0.5f + rndf(st) * (side - 1.0f);
                nt = rndf(st) * 1.5707963f;
            } else {
                nx = a.x + rndn(st) * step;
                ny = a.y + rndn(st) * step;
                nt = th[i] + rndn(st) * step * 0.9f;
            }
            float nc, ns;
            __sincosf(nt, &ns, &nc);
            float e1 = viol_of(i, nx, ny, nc, ns, q, n, side, g);
            float d = mu * (e1 - e0);
            bool acc = (d <= 0.0f) || (rndf(st) < __expf(-d / T));
            if (acc && g == 0) { q[i] = make_float4(nx, ny, nc, ns); th[i] = nt; }
            SYNC_WARP(GMASK(g));
            moves++;
        }

        for (int m = 0; m < 8; m++) {
            float w0 = wall_total(q, n, side, g);
            float dside = rndn(st) * step * 0.5f;
            float w1 = wall_total(q, n, side + dside, g);
            float d = dside + mu * (w1 - w0);
            if (d <= 0.0f || rndf(st) < __expf(-d / T)) side += dside;
        }
    }

    for (int qq = 0; qq < qstages; qq++)
        quench(q, th, side, gx, gy, gt, n, mu1 * powf(10.0f, (float)qq), 300, g);

    for (int i = g; i < n; i += LPP) {
        float4 a = q[i];
        gx2[(size_t)inst * n + i] = a.x;
        gy2[(size_t)inst * n + i] = a.y;
        gt2[(size_t)inst * n + i] = th[i];
    }
    float v = pair_total(q, n, g) + wall_total(q, n, side, g);
    if (g == 0) {
        gbest[2 * inst] = side;
        gbest[2 * inst + 1] = v;
        atomicAdd(gmoves, moves);
    }
}

static bool load_cfg(const char *path, std::vector<float> &z, int &n, double &s) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char line[512];
    n = 0; s = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            const char *p = strstr(line, "n =");
            if (p) n = atoi(p + 3);
            p = strstr(line, "s =");
            if (p) s = atof(p + 3);
            continue;
        }
        double a, b, c;
        if (sscanf(line, "%lf %lf %lf", &a, &b, &c) == 3) {
            z.push_back((float)a); z.push_back((float)b); z.push_back((float)c);
        }
    }
    fclose(f);
    n = (int)(z.size() / 3);
    return n > 0;
}

int main(int argc, char **argv) {
    const char *in = nullptr, *out = nullptr;
    int n = 300, blocks = 256, sweeps = 400, keep = 8, target = 0, qstages = 4;
    float cool = 7.0f, mstep = 0.08f, tele = 0.0f;
    float side0 = 0, perturb = 0.05f, T0 = 2e-3f, mu0 = 30.f, mu1 = 3e4f;
    unsigned seed = 12345;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&] { return argv[++i]; };
        if (a == "--in") in = nx();
        else if (a == "--out") out = nx();
        else if (a == "--n") n = atoi(nx());
        else if (a == "--blocks") blocks = atoi(nx());
        else if (a == "--sweeps") sweeps = atoi(nx());
        else if (a == "--side") side0 = atof(nx());
        else if (a == "--perturb") perturb = atof(nx());
        else if (a == "--temp") T0 = atof(nx());
        else if (a == "--mu0") mu0 = atof(nx());
        else if (a == "--mu1") mu1 = atof(nx());
        else if (a == "--seed") seed = strtoul(nx(), 0, 10);
        else if (a == "--keep") keep = atoi(nx());
        else if (a == "--target") target = atoi(nx());
        else if (a == "--quench") qstages = atoi(nx());
        else if (a == "--cool") cool = atof(nx());
        else if (a == "--tele") tele = atof(nx());
        else if (a == "--step") mstep = atof(nx());
        else { printf("unknown option %s\n", a.c_str()); return 1; }
    }

    std::vector<float> cfg;
    double rec = 0;
    if (in) {
        if (!load_cfg(in, cfg, n, rec)) { printf("cannot read %s\n", in); return 1; }
        if (side0 <= 0) side0 = (float)rec;
    }
    if (side0 <= 0) side0 = ceilf(sqrtf((float)n)) + 0.75f;

    int msrc = n;
    if (target > 0 && target < n) {
        printf("removal seeding: %d squares from the s(%d) record, keeping %d\n",
               msrc - target, msrc, target);
        n = target;
    }

    hipSetDeviceFlags(hipDeviceScheduleBlockingSync);

    int inst = blocks * PPB;
    size_t shmem = (size_t)PPB * n * (sizeof(float4) + 4 * sizeof(float));
    if (msrc > n) shmem += (size_t)PPB * msrc * sizeof(int);
    printf("ROCm/HIP MI300X: LPP=%d PPB=%d block=%d n=%d instances=%d sweeps=%d side0=%.9f shared=%zu B/block\n",
           LPP, PPB, BLOCK, n, inst, sweeps, side0, shmem);
    if (shmem > 101376) { printf("shared memory too large for a block\n"); return 1; }
    hipFuncSetAttribute((const void *)anneal, hipFuncAttributeMaxDynamicSharedMemorySize, 101376);

    float *gx, *gy, *gt, *gb, *gseed = nullptr;
    int *gkept = nullptr;
    unsigned long long *gm;
    hipMalloc(&gx, (size_t)inst * n * sizeof(float));
    hipMalloc(&gy, (size_t)inst * n * sizeof(float));
    hipMalloc(&gt, (size_t)inst * n * sizeof(float));
    hipMalloc(&gb, 2 * inst * sizeof(float));
    hipMalloc(&gm, sizeof(unsigned long long));
    hipMemset(gm, 0, sizeof(unsigned long long));
    if (msrc > n) hipMalloc(&gkept, (size_t)inst * n * sizeof(int));
    if (!cfg.empty()) {
        hipMalloc(&gseed, cfg.size() * sizeof(float));
        hipMemcpy(gseed, cfg.data(), cfg.size() * sizeof(float), hipMemcpyHostToDevice);
    }

    auto t0 = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(anneal, dim3(blocks), dim3(BLOCK), shmem, 0,
                       gseed, gx, gy, gt, gb, n, sweeps, side0, perturb,
                       T0, mu0, mu1, seed, gm, msrc, gkept, qstages,
                       cool, mstep, tele);
    hipError_t err = hipDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    if (err != hipSuccess) { printf("HIP error: %s\n", hipGetErrorString(err)); return 1; }
    double sec = std::chrono::duration<double>(t1 - t0).count();

    unsigned long long mk = 0;
    hipMemcpy(&mk, gm, sizeof mk, hipMemcpyDeviceToHost);
    std::vector<float> b(2 * inst);
    hipMemcpy(b.data(), gb, b.size() * sizeof(float), hipMemcpyDeviceToHost);
    printf("time=%.2fs  moves=%.4g  moves/s=%.4g  pairtests/s=%.4g\n",
           sec, (double)mk, mk / sec, (double)mk / sec * n);

    std::vector<int> ord(inst);
    for (int i = 0; i < inst; i++) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int c) {
        bool fa = b[2 * a + 1] < 1e-6f, fc = b[2 * c + 1] < 1e-6f;
        if (fa != fc) return fa;
        return b[2 * a] < b[2 * c];
    });
    int feas = 0;
    for (int i = 0; i < inst; i++) if (b[2 * i + 1] < 1e-6f) feas++;
    printf("near-feasible instances: %d / %d\n", feas, inst);
    for (int k = 0; k < std::min(5, inst); k++)
        printf("  #%d side=%.9f viol=%.3g\n", k, b[2 * ord[k]], b[2 * ord[k] + 1]);
    if (rec > 0 && msrc == n) printf("record=%.9f  best_gap=%+.4g\n", rec, b[2 * ord[0]] - rec);

    if (out) {
        std::vector<float> hx(n), hy(n), ht(n);
        for (int k = 0; k < std::min(keep, inst); k++) {
            int i = ord[k];
            hipMemcpy(hx.data(), gx + (size_t)i * n, n * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(hy.data(), gy + (size_t)i * n, n * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(ht.data(), gt + (size_t)i * n, n * sizeof(float), hipMemcpyDeviceToHost);
            char path[512];
            snprintf(path, sizeof path, "%s.%d.txt", out, k);
            FILE *f = fopen(path, "w");
            fprintf(f, "# n = %d\n# s = %.9f\n# x y theta(rad)\n", n, b[2 * i]);
            for (int j = 0; j < n; j++)
                fprintf(f, "%.9g %.9g %.9g\n", hx[j], hy[j], ht[j]);
            fclose(f);
        }
        printf("wrote %d candidates to %s.*.txt\n", std::min(keep, inst), out);
    }
    return 0;
}
