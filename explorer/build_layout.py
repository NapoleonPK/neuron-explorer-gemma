"""Seriated sheet layouts: places each layer's neurons on its grid (ceil(sqrt(N)) cells wide: 102 for the 10240
neurons of Gemma 4 E4B and Gemma 3 4B) so that neurons with similar weights sit together. Embeds each neuron by its write column ("writes": the
direction it adds to the residual stream) or its gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to
2D, then snaps to the grid: sort by y into rows of the grid's width, then by x within each row.

    uv run python explorer/build_layout.py --kind writes

Writes explorer/cache/layout_<kind>.npz: cell[L, N] (int16, row * 96 + col for neuron n of layer L).
"""
import argparse
import os
import sys

import numpy as np
import umap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep"))
from model_spec import MODEL, Checkpoint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL)
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
a = ap.parse_args()

ck = Checkpoint(a.model)
L, N = ck.cfg["num_hidden_layers"], ck.cfg["intermediate_size"]
side = int(np.ceil(N ** 0.5))  # grid width; the last row may be short (map3d.js uses the same width)

cells = np.zeros((L, N), np.int16)
for l in range(L):
    if a.kind == "writes":
        X = ck.write_cols(l).T.numpy()  # [N, hidden]
    else:
        X = np.concatenate([ck.tensor(l, f"mlp.{p}_proj.weight").float().numpy() for p in ("gate", "up")], 1)  # [N, 2*hidden]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, : a.pca] * S[: a.pca]
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(-(-N // side)):
        band = by_y[r * side:(r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(len(band))
    print(f"L{l} done", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", f"layout_{a.kind}.npz")
np.savez(out, cell=cells)
print("->", out)
