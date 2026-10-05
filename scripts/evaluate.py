"""Measure the quality and distinctness of the visual hash over many random names.

Usage: python scripts/evaluate.py [--count 200]

Metrics:
- lit: fraction of the image that is lit (higher = fuller image).
- white-clipped: fraction of lit pixels blown out to pure white (lower = better colour).
- chroma: average colourfulness of lit pixels (higher = richer colour).
- nearest-neighbour distance: how different each image is from the most similar other image,
  using rotation-invariant features, since the artwork spins in the UI (higher = more distinct).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import skia
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402


def random_names(count, seed=7):
    fake = Faker()
    Faker.seed(seed)
    names = []
    while len(names) < count:
        name = fake.name()
        if name not in names:
            names.append(name)
    return names


def to_rgb(png):
    return np.array(skia.Image.MakeFromEncoded(png).toarray())[::2, ::2, :3] / 255


def rgb_to_hue(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    delta = np.where(mx > mn, mx - mn, 1)
    r, g, b = np.moveaxis(rgb, -1, 0)
    hue = np.select([mx == r, mx == g], [(g - b) / delta % 6, (b - r) / delta + 2], (r - g) / delta + 4)
    return hue / 6


def features(rgb):
    """Rotation-invariant descriptor: radial brightness, angular spectrum per ring, hue histogram."""
    size = rgb.shape[0]
    center = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - center, xx - center) / center
    angle = np.arctan2(yy - center, xx - center)
    lum = rgb.mean(axis=2)

    ring = np.minimum((radius * 24).astype(int), 23).ravel()
    radial = np.bincount(ring, lum.ravel(), 24) / np.maximum(np.bincount(ring, minlength=24), 1)

    spectra = []
    for lo, hi in [(0.05, 0.2), (0.2, 0.45), (0.45, 0.7), (0.7, 0.95)]:
        mask = (radius >= lo) & (radius < hi)
        sector = ((angle[mask] + np.pi) / (2 * np.pi) * 256).astype(int) % 256
        profile = np.bincount(sector, lum[mask], 256) / np.maximum(np.bincount(sector, minlength=256), 1)
        spectrum = np.abs(np.fft.rfft(profile - profile.mean()))[1:25]
        spectra.append(spectrum / (np.linalg.norm(spectrum) + 1e-9))

    small = rgb[::4, ::4]
    chroma = small.max(-1) - small.min(-1)
    hues = np.bincount((rgb_to_hue(small) * 12).astype(int).ravel() % 12, chroma.ravel(), 12)
    hues /= hues.sum() + 1e-9

    return np.concatenate([radial / (radial.max() + 1e-9), np.concatenate(spectra) * 0.6, hues * 2.0])


def evaluate(count):
    names = random_names(count)
    start = time.time()
    images = [to_rgb(main.render_visual_hash_png(name)) for name in names]
    ms_per_image = (time.time() - start) / count * 1000

    peak = np.array([img.max(-1) for img in images])
    lit = peak > 0.12
    lit_fraction = lit.mean(axis=(1, 2))
    white = np.array([(img.min(-1) > 0.97).sum() for img in images]) / np.maximum(lit.sum(axis=(1, 2)), 1)
    chroma = np.array([(img.max(-1) - img.min(-1))[m].mean() for img, m in zip(images, lit)])

    descriptors = np.array([features(img) for img in images])
    distances = np.linalg.norm(descriptors[:, None] - descriptors[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)

    print(f"{count} names, {ms_per_image:.0f} ms/image")
    print(f"  lit            mean {lit_fraction.mean():.2f}   p5 {np.percentile(lit_fraction, 5):.2f}")
    print(f"  white-clipped  mean {white.mean() * 100:.1f}%  p95 {np.percentile(white, 95) * 100:.1f}%")
    print(f"  chroma         mean {chroma.mean():.2f}")
    print(f"  nearest-neighbour distance  median {np.median(nearest):.3f}   p5 {np.percentile(nearest, 5):.3f}")
    print(f"  most similar pair: {names[i]!r} / {names[j]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.count)
