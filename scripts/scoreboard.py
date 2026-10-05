"""Score the visual hash on everything a visual hash should be good at.

Usage: python scripts/scoreboard.py [--count 1000]

Over random names:
- distinct: how different each image is from the most similar other image (median nearest-neighbour
  distance of rotation-invariant features: shape and colour). Higher is better.
- distinct (shape): the same, from brightness alone, so colour cannot carry it. Higher is better.
- distinct at 64 px / 32 px: the same, for images shrunk to avatar size. Higher is better.
- distinct (colour-blind): the same, as someone with deuteranopia sees them. Higher is better.
- full (5th pct): how much of the canvas the least-filled 5% of images light up. Higher is better.
- white-clipped: share of lit pixels blown out to white. Lower is better.
- chroma: how colourful lit pixels are. Higher is better.
- render ms: time to draw one image the first time (median of the best of 3 runs). Lower is better.

Every measure of images comes with a 90% interval (distinctness from repeated 80% subsamples of
the names, the others by bootstrap), so a change to the algorithm can be told from noise.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import skia

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]
import evaluate  # noqa: E402
import main  # noqa: E402

# Machado, Oliveira & Fernandes (2009): deuteranopia, severity 1, on linear RGB.
DEUTERANOPIA = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])


def to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def shrink(rgb, size):
    """Area-average an image down to size x size, as a browser shows an avatar."""
    factor = rgb.shape[0] // size
    return rgb[: size * factor, : size * factor].reshape(size, factor, size, factor, 3).mean(axis=(1, 3))


def shape_features(rgb):
    """Rotation-invariant features of brightness alone: radial profile and angular spectra of 8 rings."""
    size = rgb.shape[0]
    center = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - center, xx - center) / center
    angle = np.arctan2(yy - center, xx - center)
    lum = rgb.mean(axis=2)
    ring = np.minimum((radius * 32).astype(int), 31).ravel()
    radial = np.bincount(ring, lum.ravel(), 32) / np.maximum(np.bincount(ring, minlength=32), 1)
    spectra = []
    for lo in np.linspace(0.05, 0.85, 8):
        mask = (radius >= lo) & (radius < lo + 0.1)
        sector = ((angle[mask] + np.pi) / (2 * np.pi) * 256).astype(int) % 256
        profile = np.bincount(sector, lum[mask], 256) / np.maximum(np.bincount(sector, minlength=256), 1)
        spectrum = np.abs(np.fft.rfft(profile - profile.mean()))[1:33]
        spectra.append(spectrum / (np.linalg.norm(spectrum) + 1e-9) * (profile.mean() > 0.01))
    return np.concatenate([radial / (radial.max() + 1e-9), np.concatenate(spectra) * 0.5])


def measure(name):
    png = main.render_seed_png.__wrapped__(main.name_to_seed(name))
    rgb = evaluate.to_rgb(png)  # 400 x 400, 0 to 1
    peak = rgb.max(axis=-1)
    lit = peak > 0.08
    chroma = (peak - rgb.min(axis=-1))[lit]
    colour_blind = to_srgb(to_linear(rgb) @ DEUTERANOPIA.T)
    return {
        "distinct": evaluate.features(rgb),
        "distinct (shape)": shape_features(rgb),
        "distinct at 64 px": evaluate.features(shrink(rgb, 64)),
        "distinct at 32 px": evaluate.features(shrink(rgb, 32)),
        "distinct (colour-blind)": evaluate.features(colour_blind),
        "full": lit.mean(),
        "white-clipped": (rgb.min(axis=-1)[lit] > 0.98).mean() if lit.any() else 0.0,
        "chroma": chroma.mean() if chroma.size else 0.0,
    }


def render_ms(names, rng):
    """Time to draw each name's image the first time: the best of 3 runs, one at a time (not while
    the others run). Returns (median, 90% interval)."""
    times = []
    for name in names:
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            main.render_seed_png.__wrapped__(main.name_to_seed(name))
            runs.append((time.perf_counter() - start) * 1000)
        times.append(min(runs))
    return np.median(times), np.percentile([np.median(rng.choice(times, len(times))) for _ in range(500)], [5, 95])


def nearest_distances(features, rng, rounds=200, share=0.8):
    """Median nearest-neighbour distance, and a 90% interval over subsamples of the names."""
    distance = np.linalg.norm(features[:, None] - features[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    full = np.median(distance.min(axis=1))
    estimates = []
    for _ in range(rounds):
        keep = rng.choice(len(features), int(len(features) * share), replace=False)
        estimates.append(np.median(distance[np.ix_(keep, keep)].min(axis=1)))
    return full, np.percentile(estimates, [5, 95])


def score(names, workers):
    with Pool(workers) as pool:
        results = pool.map(measure, names, chunksize=8)
    rng = np.random.default_rng(0)
    board = {}
    for key in results[0]:
        values = [result[key] for result in results]
        if key.startswith("distinct"):
            board[key] = nearest_distances(np.array(values), rng)
        else:
            statistic = (lambda v: np.percentile(v, 5)) if key == "full" else np.mean
            values = np.array(values)
            resampled = [statistic(rng.choice(values, len(values))) for _ in range(500)]
            board["full (5th pct)" if key == "full" else key] = (statistic(values), np.percentile(resampled, [5, 95]))
    board["render ms"] = render_ms(names[:40], np.random.default_rng(1))
    return board


def main_(count, workers):
    board = score(evaluate.random_names(count, seed=21), workers)
    print(f"{count} names\n")
    for key, (value, (low, high)) in board.items():
        print(f"{key:26s}{value:10.3f}   [{low:.3f}-{high:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    main_(args.count, args.workers)
