"""
Test SeverityEstimator against images in data/raw/.

The estimator uses Excess Green Index (ExG) for disease detection and
morphology-based background removal via rembg.

Usage:
  python scripts/test_severity_estimator.py            # batch: all images
  python scripts/test_severity_estimator.py <image>    # single image + live display

Outputs:
  - Per-image results printed to stdout
  - 4-panel canvas saved to data/processed/severity_<filename>
    Panels: Original | Leaf (bg removed) | ExG disease mask | Disease spots highlighted
  - Summary statistics (batch mode)
  - (single-image mode) interactive window; press any key to close
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from crop_agent.perception.severity_estimator import SeverityEstimator  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PANEL_SIZE = (400, 400)  # matches estimator's own visualize output


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def build_canvas(
    image_path: Path, estimator: SeverityEstimator, result: dict
) -> np.ndarray:
    """
    Build a 1×4 horizontal canvas identical in layout to estimate(visualize=True):
      Original | Leaf (bg removed) | ExG disease mask | Disease spots highlighted
    Labels and severity overlay are added on top.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # run pipeline to get intermediate images
    leaf, leaf_mask = estimator.remove_background(img)
    disease_mask = estimator.detect_disease(leaf, leaf_mask)
    highlighted = estimator.highlight_disease(leaf, disease_mask)

    # build ExG visualization (normalized, colour-mapped for clarity)
    leaf_float = leaf.astype("float32")
    B, G, R = cv2.split(leaf_float)
    exg = 2 * G - R - B
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    exg_color = cv2.applyColorMap(exg_norm, cv2.COLORMAP_SUMMER)
    # mask out background
    exg_color = cv2.bitwise_and(exg_color, exg_color, mask=leaf_mask)

    panels = [
        (img, "Original"),
        (leaf, "Leaf (bg removed)"),
        (exg_color, "ExG disease map"),
        (
            highlighted,
            f"{result['severity_class'].upper()}  {result['severity_percent']:.1f}%",
        ),
    ]

    out_panels = []
    for frame, label in panels:
        p = cv2.resize(frame, PANEL_SIZE)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(p, (4, 4), (14 + tw, 30), (0, 0, 0), -1)
        cv2.putText(
            p, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        out_panels.append(p)

    return np.hstack(out_panels)


def run_single(image_path: Path, estimator: SeverityEstimator):
    """Single-image mode: show live window and save canvas."""
    print(f"\nImage    : {image_path.name}")
    result = estimator.estimate(str(image_path), visualize=False)
    print(f"Severity : {result['severity_percent']:.2f}%")
    print(f"Class    : {result['severity_class']}")

    canvas = build_canvas(image_path, estimator, result)
    out_path = OUT_DIR / f"severity_{image_path.name}"
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved    : {out_path}")

    # --- show extracted leaf on its own window ---
    img = cv2.imread(str(image_path))
    leaf, _ = estimator.remove_background(img)
    leaf_resized = cv2.resize(leaf, PANEL_SIZE)
    cv2.imshow(f"Leaf Extracted — {image_path.name}", leaf_resized)

    title = f"Severity Analysis — {image_path.name}"
    cv2.imshow(title, canvas)
    print("\n[Press any key in the image window to close]\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_batch(images: list[Path], estimator: SeverityEstimator):
    """Batch mode: process all images, print table + summary, save canvases."""
    rows = []
    for img_path in images:
        print(f"  {img_path.name} ...", end=" ", flush=True)
        try:
            result = estimator.estimate(str(img_path), visualize=False)
            canvas = build_canvas(img_path, estimator, result)
            cv2.imwrite(str(OUT_DIR / f"severity_{img_path.name}"), canvas)
            rows.append(
                {
                    "image": img_path.name,
                    "severity_percent": result["severity_percent"],
                    "severity_class": result["severity_class"],
                    "ok": True,
                }
            )
            print(f"{result['severity_percent']:.1f}% ({result['severity_class']})")
        except Exception as exc:
            rows.append(
                {
                    "image": img_path.name,
                    "severity_percent": 0.0,
                    "severity_class": "—",
                    "error": str(exc),
                    "ok": False,
                }
            )
            print(f"FAILED: {exc}")

    print()
    _print_table(rows)
    _print_summary(rows)
    print(f"\nCanvases saved to: {OUT_DIR}\n")


def _print_table(rows: list[dict]):
    w = max(len(r["image"]) for r in rows)
    w = max(w, len("image"))
    header = f"{'image':<{w}}  {'severity_%':>10}  {'class':<10}  status"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        sev = f"{r['severity_percent']:.2f}" if r["ok"] else "—"
        cls = r["severity_class"] if r["ok"] else r.get("error", "error")
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['image']:<{w}}  {sev:>10}  {cls:<10}  {status}")
    print(sep)


def _print_summary(rows: list[dict]):
    ok = [r for r in rows if r["ok"]]
    if not ok:
        print("No images processed successfully.")
        return
    sevs = [r["severity_percent"] for r in ok]
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["severity_class"]] = counts.get(r["severity_class"], 0) + 1

    print(f"\nSummary  ({len(ok)}/{len(rows)} ok)")
    print(
        f"  Mean : {np.mean(sevs):.2f}%   Min : {np.min(sevs):.2f}%   Max : {np.max(sevs):.2f}%   Std : {np.std(sevs):.2f}%"
    )
    print("\n  Class distribution:")
    for cls in ["mild", "moderate", "severe", "critical"]:
        n = counts.get(cls, 0)
        print(f"    {cls:<10} {n:>3}  {'#' * n}")


def main():
    estimator = SeverityEstimator()

    if len(sys.argv) == 2:
        img_path = Path(sys.argv[1])
        if not img_path.is_absolute():
            img_path = RAW_DIR / img_path
        if not img_path.exists():
            print(f"Error: image not found: {img_path}")
            sys.exit(1)
        run_single(img_path, estimator)
        return

    images = collect_images(RAW_DIR)
    if not images:
        print(f"No images found in {RAW_DIR}")
        sys.exit(1)

    print(f"\nFound {len(images)} image(s) in {RAW_DIR}\n")
    run_batch(images, estimator)


if __name__ == "__main__":
    main()
