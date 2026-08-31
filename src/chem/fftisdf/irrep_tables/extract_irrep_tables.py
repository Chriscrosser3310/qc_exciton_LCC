# Same parsers as jsun3's extract_irrep_tables.py, pointed at our c8 logs.
import glob, os, re, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def parse_table(lines, title, stops):
    rows, inside = [], False
    for line in lines:
        if line.strip() == title: inside = True; continue
        if not inside: continue
        s = line.strip()
        if not s: continue
        if any(s.startswith(p) for p in stops): break
        f = s.split()
        if f[0] in ["irrep_dim","W","AO","X"]: continue
        try: rows.append([int(x) for x in f[:3]])
        except ValueError: pass
    return np.asarray(rows, dtype=int)
def parse_grid_ao(lines):
    rows, inside = [], False
    for line in lines:
        if line.strip() == "X allowed blocks: selected-grid irreps against AO irreps":
            inside = True; continue
        if not inside: continue
        s = line.strip()
        if not s: continue
        f = s.split()
        if f[0] == "irrep_dim": continue
        try: rows.append([int(f[0]), int(f[1]), int(f[2])])
        except (ValueError, IndexError): pass
    return np.asarray(rows, dtype=int)
for lf in sorted(glob.glob("irreps_ov_*_c8_final.log")):
    lines = open(lf).readlines()
    meta = {}
    for line in lines:
        if line.startswith("kmesh ="): meta['k'] = "x".join(re.findall(r"\d+", line)[:3])
        elif line.startswith("n selected points ="): meta['nIP'] = int(line.split("=")[1])
    if 'k' not in meta: print("skip", lf); continue
    tabs = {"grid-grid": parse_table(lines, "W / selected-grid block structure", ["W dimension check"]),
            "ao-ao":     parse_table(lines, "AO/orbital block structure", ["AO dimension check"]),
            "grid-ao":   parse_grid_ao(lines)}
    for pat, dat in tabs.items():
        out = f"data_ov_diamond_{meta['k']}_c8_{pat}.txt"
        np.savetxt(out, dat, fmt="%d")
        print(f"{out:44s} {str(dat.shape):>10s}   nIP={meta['nIP']}")
