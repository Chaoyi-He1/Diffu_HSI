"""Cluster-robust standard errors over source images, shared by the evaluation harness and the Phase 1a gate.

The gate cubes are patches of source images (the 200-cube subset holds 113 images, 1-5 patches each), so patches of
one image are not independent and a cube-level SE understates the uncertainty of a mean over cubes. These SEs are
reported next to the cube-level ones; the gate's k-SE rule keeps the cube-level SE (spec §8). numpy only, so the
gate stays free of torch.
"""
import os

import numpy as np


def source_image(path):
    """Cluster key of a cube file: its class directory plus the image stem (basename.rsplit('_', 1)[0]),
    e.g. MatFlower60/Train/P023/34_10.mat -> 'P023/34'. Relative and absolute paths give the same key."""
    return os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path).rsplit('_', 1)[0]


def cluster_se(d, files):
    """CR1 cluster-robust SE of mean(d), clustered by source_image(files):

        sqrt(G / (G - 1) * sum_g (sum_{i in g} (d_i - mean(d)))^2) / n

    With one cube per image this equals the iid SE std(d, ddof=1) / sqrt(n). Returns (se, G); se is None when the
    cubes come from fewer than 2 images."""
    d = np.asarray(d, dtype=np.float64)
    if len(files) != len(d):
        raise ValueError(f'{len(d)} values but {len(files)} files')
    keys = [source_image(f) for f in files]
    pos = {k: i for i, k in enumerate(sorted(set(keys)))}
    G = len(pos)
    if G < 2:
        return None, G
    sums = np.bincount([pos[k] for k in keys], weights=d - d.mean(), minlength=G)   # per-image sums of d_i - mean(d)
    return float(np.sqrt(G / (G - 1) * np.sum(sums ** 2)) / len(d)), G
