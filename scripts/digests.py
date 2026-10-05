"""Print a digest of the pixels of a few names' images, to check that every platform draws them alike.

Usage: python scripts/digests.py

Compare its output across machines, e.g. this Mac against the Docker image on both architectures:

    python scripts/digests.py > here.txt
    for platform in linux/arm64 linux/amd64; do
      docker build -q --platform $platform -t visual-hashing:check . &&
      docker run --rm --platform $platform -v "$PWD/scripts:/app/scripts:ro" visual-hashing:check \\
        python scripts/digests.py | diff here.txt - && echo "$platform: identical"
    done
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import skia

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

NAMES = ["Budi", "Ani", "Siti Rahma", "李小龙 🐉", "Gun Gun Feb", "Alice", "Purwo", "Maria Garcia"]

if __name__ == "__main__":
    for name in NAMES:
        png = main.render_visual_hash_png(name)
        pixels = np.array(skia.Image.MakeFromEncoded(png).toarray(colorType=skia.kRGBA_8888_ColorType))
        print(main.hash_id(name), name, hashlib.sha256(pixels.tobytes()).hexdigest()[:16])
