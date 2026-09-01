#!/usr/bin/env python3
"""Build new locket expression stills from the original calibrated set.

The live renderer swaps source portraits above a strength threshold. Those
portraits must share the same crop, head size and hand pose as the August
xiaoya_locket references. This script never regenerates a full figure; it
only blends eyes / brows / mouth from the locked originals, plus a local
eyelid droop for sleepy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPR_DIR = ROOT / "assets" / "expressions" / "xiaoya_locket"
CANDIDATE_DIR = ROOT / "assets" / "expressions" / "candidates" / "stills"

# Pixel boxes on the 941x1672 August stills (see the y-grid on smirk).
REGIONS = {
    "face": (200, 280, 760, 1040),
    "brows": (270, 330, 690, 500),
    "eyes": (260, 460, 700, 690),
    "eye_l": (270, 460, 475, 690),
    "eye_r": (485, 460, 700, 690),
    "mouth": (320, 740, 660, 960),
    "cheeks": (250, 620, 730, 900),
}

ORIGINALS = (
    "reference-laugh.png",
    "reference-smirk.png",
    "reference-shy.png",
    "reference-wink.png",
    "reference-pout.png",
    "reference-one-brow.png",
    "reference-surprised.png",
    "reference-cute-annoyed.png",
    "reference-cheek-puff.png",
)


def load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != (1672, 941):
        raise SystemExit(f"{path} must be 941x1672, got {image.shape[1]}x{image.shape[0]}")
    return image


def ellipse_mask(height: int, width: int, feather: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - (width - 1) / 2) / max(1.0, (width - 1) / 2)
    ny = (yy - (height - 1) / 2) / max(1.0, (height - 1) / 2)
    radius = np.sqrt(nx * nx + ny * ny)
    inner = 1.0 - min(0.85, max(0.08, feather))
    alpha = np.clip((1.0 - radius) / max(1e-6, 1.0 - inner), 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def blend_region(
    dest: np.ndarray,
    source: np.ndarray,
    name: str,
    strength: float,
    feather: float = 0.42,
) -> None:
    if strength <= 0:
        return
    x0, y0, x1, y1 = REGIONS[name]
    patch_d = dest[y0:y1, x0:x1].astype(np.float32)
    patch_s = source[y0:y1, x0:x1].astype(np.float32)
    mask = ellipse_mask(y1 - y0, x1 - x0, feather)[..., None] * float(strength)
    dest[y0:y1, x0:x1] = np.clip(patch_d * (1.0 - mask) + patch_s * mask, 0, 255).astype(
        np.uint8
    )


def droop_eyes(image: np.ndarray, amount: float = 0.90) -> np.ndarray:
    """Close both lids using pixels already on this portrait."""

    out = image.copy()
    for name in ("eye_l", "eye_r"):
        x0, y0, x1, y1 = REGIONS[name]
        eye = out[y0:y1, x0:x1].astype(np.float32)
        height, width = eye.shape[:2]
        lid_h = max(8, int(height * 0.44))
        stretched = cv2.resize(eye[:lid_h], (width, height), interpolation=cv2.INTER_CUBIC)
        yy = np.linspace(0.0, 1.0, height)[:, None, None]
        xx = np.linspace(-1.0, 1.0, width)[None, :, None]
        horiz = np.clip(1.0 - xx * xx, 0.0, 1.0)
        vert = np.clip(1.0 - yy * 0.48, 0.0, 1.0)
        edge = ellipse_mask(height, width, 0.24)[..., None]
        alpha = amount * horiz * vert * edge
        out[y0:y1, x0:x1] = np.clip(eye * (1.0 - alpha) + stretched * alpha, 0, 255).astype(
            np.uint8
        )
    return out


def compose() -> dict[str, np.ndarray]:
    src = {name.removeprefix("reference-").removesuffix(".png"): load_rgb(EXPR_DIR / name) for name in ORIGINALS}

    soft_smile = src["smirk"].copy()
    blend_region(soft_smile, src["laugh"], "eyes", 0.42, feather=0.38)
    blend_region(soft_smile, src["laugh"], "cheeks", 0.18, feather=0.50)

    curious = src["one-brow"].copy()
    blend_region(curious, src["surprised"], "eyes", 0.80, feather=0.34)
    blend_region(curious, src["surprised"], "brows", 0.28, feather=0.38)
    blend_region(curious, src["surprised"], "mouth", 0.20, feather=0.45)

    # Hard swap sources stay on a single August plate so identity cannot drift.
    # Shy already looks aside; pout is the only safe closed-mouth bite stand-in.
    side_eye = src["shy"].copy()
    lip_bite = src["pout"].copy()
    sleepy = droop_eyes(src["one-brow"].copy(), amount=0.70)

    tender = src["shy"].copy()
    blend_region(tender, src["pout"], "mouth", 0.48, feather=0.40)
    blend_region(tender, src["cute-annoyed"], "brows", 0.18, feather=0.42)

    return {
        "soft-smile": soft_smile,
        "curious": curious,
        "side-eye": side_eye,
        "lip-bite": lip_bite,
        "sleepy": sleepy,
        "tender": tender,
    }


def body_mse(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    body = (a[int(h * 0.48) : int(h * 0.98), int(w * 0.10) : int(w * 0.90)].astype(np.float32)
            - b[int(h * 0.48) : int(h * 0.98), int(w * 0.10) : int(w * 0.90)].astype(np.float32))
    return float((body * body).mean())


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite live reference PNGs")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/locket-composed"))
    args = parser.parse_args()

    built = compose()
    laugh = load_rgb(EXPR_DIR / "reference-laugh.png")
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'name':12s} {'body_mse':>8s}  path")
    for name, image in built.items():
        mse = body_mse(image, laugh)
        preview = args.preview_dir / f"{name}.png"
        write_png(preview, image)
        face = image[280:1040, 200:760]
        Image.fromarray(face).resize((320, 430)).save(args.preview_dir / f"face-{name}.jpg", quality=90)
        print(f"{name:12s} {mse:8.1f}  {preview}")
        if args.write:
            live = EXPR_DIR / f"reference-{name}.png"
            write_png(live, image)
            candidate = CANDIDATE_DIR / f"{name}.png"
            if candidate.parent.is_dir():
                write_png(candidate, image)
            print(f"  wrote {live}")


if __name__ == "__main__":
    main()
