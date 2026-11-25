"""Generate the bundled sample photo set (5 grayscale PNGs + annotation hint).

Run:  python scripts/generate_samples.py

The set simulates five same-pose shots of a mural wall with:
  * a shared, mostly flat calibration strip (top band),
  * per-image exposure offsets,
  * per-image scaffolding occlusion rectangles,
  * one region covered by enough boxes that it must be flagged 待补拍.

Everything is deterministic (fixed RNG seed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20261125
SIZE = 256
CALIB = (32, 16, 224, 48)  # x0, y0, x1, y1


def mural_base() -> np.ndarray:
    """Synthetic 256x256 'mural' wall, uint8."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    # Soft vertical illumination gradient plus broad brush bands.
    base = 120 + 35 * np.sin(xx / 38.0) + 25 * np.cos(yy / 55.0)
    bands = 20 * np.sin((xx + yy) / 23.0)
    texture = rng.normal(0, 4, size=(SIZE, SIZE))
    img = base + bands + texture
    # Keep the calibration band itself low-noise and safely unsaturated.
    x0, y0, x1, y1 = CALIB
    cal = 128 + 6 * np.sin(xx[y0:y1, x0:x1] / 30.0) + rng.normal(0, 1.5, (y1 - y0, x1 - x0))
    img[y0:y1, x0:x1] = cal
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def shift_and_occlude(base: np.ndarray, offset: float, box, rng) -> np.ndarray:
    img = base.astype(np.float64) + offset
    img += rng.normal(0, 2.0, size=img.shape)  # small shot noise
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        # Scaffolding: bright poles with slight shading.
        img[y0:y1, x0:x1] = rng.integers(190, 230, size=(y1 - y0, x1 - x0), dtype=np.int64)
    return img


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = mural_base()
    rng = np.random.default_rng(SEED + 1)

    # (exposure offset, occlusion box). The center patch (96,120)-(150,200)
    # is covered in images 0, 2, 3, 4 simultaneously, leaving only image 1
    # to contribute -> flagged 待补拍. Image 0's scaffolding also covers a
    # right-side pole, image 1's a left-side pole.
    configs = [
        (0.0, (96, 120, 150, 200)),    # image 0: baseline, center patch
        (-14.0, (40, 70, 66, 230)),    # image 1: darker, pole on left
        (22.0, (96, 120, 150, 200)),   # image 2: brighter, center patch
        (-7.5, (96, 120, 150, 200)),   # image 3: same center patch
        (11.0, (96, 120, 150, 200)),   # image 4: same center patch
    ]
    boxes = []
    for i, (offset, box) in enumerate(configs):
        img = shift_and_occlude(base, offset, box, rng)
        Image.fromarray(img, mode="L").save(out_dir / f"sample_{i}.png", format="PNG")
        boxes.append(list(box))

    meta = {"calib": list(CALIB), "boxes": boxes}
    (out_dir / "annotation.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(configs)} sample images and annotation.json to {out_dir}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "samples"
    main(target)
