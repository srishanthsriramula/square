// High-performance Fault-Line Gadget DP Solver in C++17
// Computes optimal multi-gadget compositions across all (K, D) pairs in milliseconds.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

constexpr int MAX_K = 32;
constexpr int MAX_D = 120;
constexpr int MAX_DEPTH = 12;
constexpr double INF = 1e9;

struct Gadget {
    int id;
    int w;
    int h;
    int dm;
    double f;
    std::string name;
};

struct DPState {
    double f = INF;
    int parent_c = -1;
    int parent_r = -1;
    int parent_d = -1;
    int gadget_idx = -1;
    int depth = 0;
};

int main(int argc, char** argv) {
    std::string input_file = "data/generated/gadgets.txt";
    if (argc > 1) {
        input_file = argv[1];
    }

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "Could not open gadgets file: " << input_file << "\n";
        return 1;
    }

    std::vector<Gadget> gadgets;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Gadget g;
        if (iss >> g.id >> g.w >> g.h >> g.dm >> g.f >> g.name) {
            if (g.w <= MAX_K && g.h <= MAX_K && g.dm > 0 && g.f < 1.0f) {
                gadgets.push_back(g);
            }
        }
    }
    std::cerr << "Loaded " << gadgets.size() << " valid gadgets.\n";

    // Results table for each (K, D)
    struct BestKD {
        double min_f = INF;
        std::vector<int> chain;
    };
    BestKD best_kd[MAX_K + 1][MAX_D + 1];

    // Run DP for each container integer size K
    for (int K = 2; K <= MAX_K; ++K) {
        // Flat DP table for fixed K: dp[c][r][d]
        std::vector<std::vector<std::vector<DPState>>> dp(
            K + 1, std::vector<std::vector<DPState>>(
                K + 1, std::vector<DPState>(MAX_D + 1)
            )
        );

        dp[0][0][0].f = 0.0f;

        // Filter gadgets fitting in K x K
        std::vector<int> cand_indices;
        for (size_t i = 0; i < gadgets.size(); ++i) {
            if (gadgets[i].w <= K && gadgets[i].h <= K) {
                cand_indices.push_back(i);
            }
        }

        // DP rounds (chain depth up to MAX_DEPTH)
        for (int depth = 0; depth < MAX_DEPTH; ++depth) {
            for (int c = 0; c <= K; ++c) {
                for (int r = 0; r <= K; ++r) {
                    for (int d = 0; d <= MAX_D; ++d) {
                        double cur_f = dp[c][r][d].f;
                        if (cur_f >= INF) continue;
                        if (dp[c][r][d].depth != depth) continue;

                        for (int g_idx : cand_indices) {
                            const auto& g = gadgets[g_idx];
                            int new_c = c + g.w + (depth > 0 ? 1 : 0);
                            int new_r = r + g.h + (depth > 0 ? 1 : 0);
                            int new_d = d + g.dm;
                            double new_f = std::max(cur_f, g.f);

                            if (new_c <= K && new_r <= K && new_d <= MAX_D && new_f < 0.999999) {
                                if (new_f < dp[new_c][new_r][new_d].f) {
                                    dp[new_c][new_r][new_d].f = new_f;
                                    dp[new_c][new_r][new_d].parent_c = c;
                                    dp[new_c][new_r][new_d].parent_r = r;
                                    dp[new_c][new_r][new_d].parent_d = d;
                                    dp[new_c][new_r][new_d].gadget_idx = g_idx;
                                    dp[new_c][new_r][new_d].depth = depth + 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Extract best for each D
        for (int c = 0; c <= K; ++c) {
            for (int r = 0; r <= K; ++r) {
                for (int d = 1; d <= MAX_D; ++d) {
                    if (dp[c][r][d].f < best_kd[K][d].min_f) {
                        best_kd[K][d].min_f = dp[c][r][d].f;
                        // Reconstruct chain
                        std::vector<int> chain;
                        int curr_c = c, curr_r = r, curr_d = d;
                        while (curr_d > 0) {
                            const auto& st = dp[curr_c][curr_r][curr_d];
                            chain.push_back(st.gadget_idx);
                            curr_c = st.parent_c;
                            curr_r = st.parent_r;
                            curr_d = st.parent_d;
                        }
                        std::reverse(chain.begin(), chain.end());
                        best_kd[K][d].chain = chain;
                    }
                }
            }
        }
    }

    // Output results as CSV: K,D,f,chain_length,gadget_ids
    std::cout << "K,D,f,chain_len,gadget_names\n";
    for (int K = 2; K <= MAX_K; ++K) {
        for (int d = 1; d <= MAX_D; ++d) {
            if (best_kd[K][d].min_f < INF) {
                std::cout << K << "," << d << "," << std::fixed << std::setprecision(16)
                          << best_kd[K][d].min_f << "," << best_kd[K][d].chain.size() << ",\"";
                for (size_t i = 0; i < best_kd[K][d].chain.size(); ++i) {
                    if (i > 0) std::cout << ";";
                    std::cout << gadgets[best_kd[K][d].chain[i]].name;
                }
                std::cout << "\"\n";
            }
        }
    }

    return 0;
}
