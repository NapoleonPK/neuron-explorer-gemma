"""Stacks the per-layer steer lens files from sweep/steer_lens.py into the explorer's steer index.

    uv run python explorer/build_steer_index.py --stats stats/steer_lens

Reads stats/steer_lens_L{l}.npz for every layer (all must exist) and writes explorer/cache/steer_index.npz:
up[L, N, k], down[L, N, k] (int32 token ids), up_kl[L, N], down_kl[L, N] (float16).
"""
import argparse
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stats", default="stats/steer_lens", help="prefix given to steer_lens.py --out")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "steer_index.npz"))
a = ap.parse_args()

paths = {int(re.search(r"_L(\d+)\.npz$", p).group(1)): p for p in glob.glob(f"{a.stats}_L*.npz")}
L = max(paths) + 1
missing = sorted(set(range(L)) - set(paths))
assert not missing, f"steer lens missing for layers {missing}"
z = [np.load(paths[l]) for l in range(L)]
np.savez(a.out, up=np.stack([x["up_tok"] for x in z]).astype(np.int32),
         down=np.stack([x["down_tok"] for x in z]).astype(np.int32),
         up_kl=np.stack([x["up_kl"] for x in z]).astype(np.float16),
         down_kl=np.stack([x["down_kl"] for x in z]).astype(np.float16))
print(f"{L} layers -> {a.out}")
