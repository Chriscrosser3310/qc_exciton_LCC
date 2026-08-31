#!/usr/bin/env python
"""Compare a reproduced optimized checkpoint against jsun3's shipped one.

Reports three things, in increasing order of how much they mean:

1. **Individual factors** (``||X^o||_op`` etc.). These are expected to disagree, and a
   disagreement here is not a failure -- see 2.
2. **The scale gauge.** ``X -> s X`` with ``W -> W / s^4`` leaves both the represented
   tensor and every alpha norm exactly invariant, because the reconstruction is quartic
   in X and linear in W, and W is re-solved by least squares for whatever X it is given.
   The loss is therefore flat along this direction and Adam wanders along it freely. Two
   runs can land far apart in ``||X||`` while being the same factorization.
3. **The gauge-invariant spectrum** of the Q=0 ov block, and the alpha norms. These are
   what actually has to match.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from chem.lcu_norms import compute_norms, load_thc_factors  # noqa: E402

CHK = "ISDFov_opt_bareGDF_2x2x2_c5_cref12_norm0.1p2.chk"
PKL = "DFT_2x2x2.pkl"


def spectrum(x_occ, x_vir, coul):
    """Singular values of the Q=0 ov block, via the M x M reduction.

    Invariant under per-k unitary mixing within occ and within vir, and under the scale
    gauge, so it compares the physics rather than the parametrization.
    """
    w0 = np.asarray(coul[0]).astype(np.complex128)
    n_k, n_aux, _ = x_occ.shape
    gram_l = np.zeros((n_aux, n_aux), dtype=np.complex128)
    gram_r = np.zeros((n_aux, n_aux), dtype=np.complex128)
    for k in range(n_k):
        occ = x_occ[k] @ x_occ[k].conj().T
        vir = x_vir[k] @ x_vir[k].conj().T
        gram_l += occ * vir.conj()
        gram_r += occ.conj() * vir
    eig = np.linalg.eigvals((w0.conj().T @ gram_l @ w0) @ gram_r.conj())
    return np.sqrt(np.sort(np.abs(eig))[::-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsun3", required=True, help="directory holding the shipped checkpoint")
    parser.add_argument("--repro", required=True, help="directory holding the reproduction")
    parser.add_argument("--chk", default=CHK)
    args = parser.parse_args(argv)

    loaded = {}
    for label, directory in (("jsun3", args.jsun3), ("repro", args.repro)):
        path = os.path.join(directory, args.chk)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        loaded[label] = load_thc_factors(path, dft_pkl=os.path.join(directory, PKL))

    norms = {k: compute_norms(*v) for k, v in loaded.items()}
    a, b = norms["jsun3"], norms["repro"]

    print("1. individual factors -- differences here are the scale gauge, not error")
    print(f"   {'quantity':18s}{'jsun3':>15s}{'repro':>15s}{'ratio':>10s}")
    for key in ("xo_op", "xv_op", "xo_2inf", "xv_2inf", "w_op_max", "w_fro_max"):
        x, y = getattr(a, key), getattr(b, key)
        print(f"   {key:18s}{x:15.6g}{y:15.6g}{y / x:10.4f}")

    print("\n2. scale-gauge check: X -> s X implies W -> W / s^4")
    s_occ = np.linalg.norm(loaded["repro"][0]) / np.linalg.norm(loaded["jsun3"][0])
    s_vir = np.linalg.norm(loaded["repro"][1]) / np.linalg.norm(loaded["jsun3"][1])
    w_ratio = np.linalg.norm(loaded["repro"][2]) / np.linalg.norm(loaded["jsun3"][2])
    print(f"   s from ||X^o|| = {s_occ:.4f},  s from ||X^v|| = {s_vir:.4f}")
    print(f"   ||W|| ratio    = {w_ratio:.4f}   predicted 1/(s_o^2 s_v^2) = "
          f"{1.0 / (s_occ ** 2 * s_vir ** 2):.4f}")

    print("\n3. what has to match")
    print(f"   {'quantity':22s}{'jsun3':>15s}{'repro':>15s}{'rel diff':>12s}")
    for key in ("alpha_embed_op", "alpha_embed_op_mod", "alpha_lcu_dir", "alpha_lcu_exch"):
        x, y = getattr(a, key), getattr(b, key)
        print(f"   {key:22s}{x:15.6g}{y:15.6g}{abs(y - x) / abs(x):12.3e}")

    spec_a = spectrum(*loaded["jsun3"][:3])
    spec_b = spectrum(*loaded["repro"][:3])
    n = min(len(spec_a), len(spec_b))
    dev = np.linalg.norm(spec_a[:n] - spec_b[:n]) / np.linalg.norm(spec_a[:n])
    print(f"\n   gauge-invariant Q=0 spectrum, top 6:")
    print(f"     jsun3 {np.array2string(spec_a[:6], precision=6)}")
    print(f"     repro {np.array2string(spec_b[:6], precision=6)}")
    print(f"   rel L2 deviation over {n} singular values = {dev:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
