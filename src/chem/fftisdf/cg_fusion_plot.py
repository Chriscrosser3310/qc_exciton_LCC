"""Plot the diamond CG fusion tensor  C^{lambda3 alpha3}_{alpha1 alpha2}  (r5 (x) r5 = r1 (+) r5 (+) r18).

A 3-way tensor is shown the standard way: as mode-3 slices -- one matrix per output basis state.
Here each slice is a 2x2 matrix over the two input partner indices (alpha1, alpha2). Entries are
complex, so each cell is domain-colored: hue = phase, brightness = |coefficient|; cells are also
annotated with |C| and the phase in degrees.  Reads only the saved npz (no pyscf needed).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(OUT_DIR, "cg_coefficients_diamond_2x2x2.npz"), allow_pickle=True)
V = d["V"]                       # (4, 4): rows (a1,a2)=00,01,10,11 ; cols = output states
labels = d["labels"]             # (4, 2): (lambda3, alpha3)
dim = d["dim"]
nout = V.shape[1]
C = V.reshape(2, 2, nout)        # 3-way tensor  C[alpha1, alpha2, out]
vmax = np.abs(V).max()

def c2rgb(Z):
    H = (np.angle(Z) % (2 * np.pi)) / (2 * np.pi)
    S = np.ones_like(H)
    Val = np.clip(np.abs(Z) / vmax, 0, 1)
    return mcolors.hsv_to_rgb(np.stack([H, S, Val], -1))

out_name = []
for (lam, al) in labels:
    if dim[lam] > 1:
        out_name.append(f"→ r{lam}  (dim {dim[lam]}), α₃={al}")
    else:
        out_name.append(f"→ r{lam}  (singlet)")

fig = plt.figure(figsize=(15, 4.2))
gs = fig.add_gridspec(1, nout + 1, width_ratios=[1] * nout + [0.6])

for b in range(nout):
    ax = fig.add_subplot(gs[0, b])
    ax.imshow(c2rgb(C[:, :, b]), origin="upper")
    for i in range(2):
        for j in range(2):
            z = C[i, j, b]
            if abs(z) > 0.02:
                txt = f"{abs(z):.2f}\n∠{np.degrees(np.angle(z)):.0f}°"
                col = "white" if abs(z) / vmax < 0.6 else "black"
            else:
                txt, col = "0", "#888888"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["α₂=0", "α₂=1"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["α₁=0", "α₁=1"], fontsize=9)
    ax.set_title(out_name[b], fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)

# phase wheel legend
axw = fig.add_subplot(gs[0, nout], projection="polar")
th = np.linspace(0, 2 * np.pi, 361)
rr = np.linspace(0, 1, 40)
TH, RR = np.meshgrid(th, rr)
axw.pcolormesh(TH, RR, TH, cmap="hsv", shading="auto")
axw.set_yticklabels([]); axw.set_xticks(np.deg2rad([0, 90, 180, 270]))
axw.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=8)
axw.set_title("phase → hue\nradius → |C|", fontsize=9, pad=12)

fig.suptitle("Diamond CG fusion tensor  C^{λ₃α₃}_{α₁α₂}   for   r₅ ⊗ r₅ = r₁ ⊕ r₅ ⊕ r₁₈\n"
             "(mode-3 slices: one 2×2 input matrix per output state;  |C|=1/√2 on every coupling)",
             fontsize=11)
fig.text(0.5, -0.02,
         "Gauge note: the overall phase of each output block is a convention (here ≈57°); "
         "the |C| pattern and within-block relative phases (e.g. the +/− in r₁₈ = antisymmetric singlet) are physical.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.02, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"cg_fusion_tensor.{ext}"), dpi=150, bbox_inches="tight")
print("saved cg_fusion_tensor.{png,pdf}")
