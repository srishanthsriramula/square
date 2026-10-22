import urllib.request
import re
import pandas as pd

url = 'https://kingbird.myphotos.cc/packing/squares_in_squares.html'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')

parts = html.split('<div class="box">')[1:]
print(f'Total box parts: {len(parts)}')

ellsworth = {}
for p in parts:
    m_n = re.search(r'<font size="\+3">([^<]+)<br>', p)
    if not m_n:
        continue
    raw_ns = m_n.group(1).replace('&thinsp;', '').replace('&nbsp;', '')
    ns = [int(x.strip()) for x in raw_ns.split(',') if x.strip().isdigit()]
    
    # search for "$s = "
    m_s = re.search(r'\$s\s*=\s*([^$]+)\$', p)
    if not m_s:
        continue
    s_str = m_s.group(1)
    
    m_dec = re.search(r'\\Nn\{([0-9\.]+)\}', s_str)
    if m_dec:
        s_val = float(m_dec.group(1))
    else:
        m_num = re.search(r'([0-9]+\.[0-9]+)', s_str)
        if m_num:
            s_val = float(m_num.group(1))
        else:
            m_int = re.search(r'=\s*([0-9]+)', s_str)
            if not m_int:
                m_int = re.search(r'([0-9]+)', s_str)
            s_val = float(m_int.group(1)) if m_int else None
            
    if s_val is not None:
        for n in ns:
            ellsworth[n] = s_val

print(f'Successfully parsed {len(ellsworth)} instances from Ellsworth.')

df_cur = pd.read_csv('results/score_by_n.csv')
cur_dict = dict(zip(df_cur['n'], df_cur['L']))

better_in_ellsworth = []
better_in_ours = []
equal = []

for n, s_e in sorted(ellsworth.items()):
    if n in cur_dict:
        s_c = cur_dict[n]
        diff = s_c - s_e
        if diff > 1e-5:
            better_in_ellsworth.append((n, s_c, s_e, diff))
        elif diff < -1e-5:
            better_in_ours.append((n, s_c, s_e, -diff))
        else:
            equal.append(n)

print(f'Comparison:')
print(f'  Ellsworth strictly better: {len(better_in_ellsworth)}')
print(f'  Ours strictly better:      {len(better_in_ours)}')
print(f'  Equal (within 1e-5):       {len(equal)}')

if better_in_ellsworth:
    print('\nInstances where Ellsworth is strictly better:')
    for n, s_c, s_e, diff in sorted(better_in_ellsworth, key=lambda x: x[3], reverse=True)[:35]:
        print(f'  N={n:3d}: Ours={s_c:.6f}, Ellsworth={s_e:.6f}, Difference={diff:.6f}')
