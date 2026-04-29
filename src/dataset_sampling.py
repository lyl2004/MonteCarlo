from __future__ import annotations

import random
from typing import Any


def lhs_samples(ranges: dict[str, list[float]], n_samples: int, seed: int = 0) -> list[dict[str, float]]:
    """Small Latin-hypercube sampler for scalar numeric ranges."""
    if n_samples <= 0:
        return []
    rng = random.Random(seed)
    columns: dict[str, list[float]] = {}
    for name, bounds in ranges.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        if hi < lo:
            lo, hi = hi, lo
        vals = [lo + (hi - lo) * ((i + rng.random()) / n_samples) for i in range(n_samples)]
        rng.shuffle(vals)
        columns[name] = vals
    return [{name: vals[i] for name, vals in columns.items()} for i in range(n_samples)]


def grid_or_constant_params(spec: dict[str, Any], n_samples: int, seed: int = 0) -> list[dict[str, Any]]:
    ranges = {k: v for k, v in spec.items() if isinstance(v, list) and len(v) == 2}
    constants = {k: v for k, v in spec.items() if k not in ranges}
    samples = lhs_samples(ranges, n_samples, seed=seed)
    for sample in samples:
        sample.update(constants)
    return samples
