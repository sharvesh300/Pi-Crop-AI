"""
Remove background from a single leaf image and show the result.

Usage:
  python scripts/remove_bg.py <image_path>
  python scripts/remove_bg.py            # picks first image in data/raw/

Shows a side-by-side dialog: Original | Leaf (background removed)
Press any key in the window to close.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from rembg import remove

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DISPLAY_SIZE = (480, 480)


def remove_background(img: np.ndarray):
    """Remove background using rembg + morphological cleanup."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = remove(rgb)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGRA)

    alpha = output[:, :, 3]
    mask = np.where(alpha > 0, 255, 0).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    leaf = cv2.bitwise_and(img, img, mask=mask)
    return leaf, mask


def resolve_image(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            sys.exit(f"[ERROR] File not found: {p}")
        return p

    raw_dir = ROOT / "data" / "raw"
    candidates = sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not candidates:
        sys.exit(f"[ERROR] No images found in {raw_dir}")
    return candidates[0]


def add_label(panel: np.ndarray, text: str) -> np.ndarray:
    p = panel.copy()
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(p, (4, 4), (16 + tw, 34), (0, 0, 0), -1)
    cv2.putText(p, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return p


def main():
    image_path = resolve_image(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Image  : {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"[ERROR] Could not read image: {image_path}")

    print("Removing background ...")
    leaf, mask = remove_background(img)
    print("Done.")

    # build side-by-side panels
    orig_panel = add_label(cv2.resize(img, DISPLAY_SIZE), "Original")
    leaf_panel = add_label(cv2.resize(leaf, DISPLAY_SIZE), "Leaf (bg removed)")
    canvas = np.hstack([orig_panel, leaf_panel])

    window_title = f"Background Removal — {image_path.name}"
    cv2.imshow(window_title, canvas)
    print("\n[Press any key in the image window to close]\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
