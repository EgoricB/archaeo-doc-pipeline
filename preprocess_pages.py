from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from skimage.filters import threshold_otsu
from skimage.transform import rotate


PAGES_DIR = Path("pages")
PROCESSED_DIR = Path("processed_pages")
MANIFEST_PATH = Path("processed_pages_manifest.csv")

VALID_EXTS = {".png", ".jpg", ".jpeg"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_page_images():
    if not PAGES_DIR.exists():
        print(f"ERROR: pages directory not found: {PAGES_DIR}")
        return

    for report_dir in sorted(PAGES_DIR.iterdir()):
        if not report_dir.is_dir():
            continue

        for img_path in sorted(report_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXTS:
                yield report_dir.name, img_path


def pil_to_gray(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def estimate_skew_angle(gray: Image.Image) -> float:
    arr = np.array(gray)

    bw = arr < np.percentile(arr, 60)

    best_angle = 0.0
    best_score = -1.0

    for angle in np.linspace(-3, 3, 25):
        rot = rotate(
            bw.astype(float),
            angle=angle,
            resize=False,
            mode="edge",
            preserve_range=True,
        )
        proj = rot.sum(axis=1)
        score = proj.var()

        if score > best_score:
            best_score = score
            best_angle = angle

    return float(best_angle)


def preprocess_image(image_path: Path, output_path: Path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    gray = pil_to_gray(img)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))

    angle = estimate_skew_angle(gray)

    deskewed = gray.rotate(
        angle,
        expand=True,
        fillcolor=255
    )

    arr = np.array(deskewed)
    thr = threshold_otsu(arr)
    bw = (arr > thr).astype(np.uint8) * 255

    out = Image.fromarray(bw)

    ensure_dir(output_path.parent)
    out.save(output_path)

    return {
        "processed_image_path": str(output_path.resolve()),
        "estimated_skew_angle": round(angle, 4),
        "threshold_otsu": int(thr),
        "width": out.width,
        "height": out.height,
    }


def main():
    ensure_dir(PROCESSED_DIR)

    records = []
    total = 0
    errors = 0

    for report_id, img_path in iter_page_images() or []:
        total += 1

        out_dir = PROCESSED_DIR / report_id
        out_path = out_dir / img_path.name

        print(f"[{total}] Processing: {img_path}")

        try:
            meta = preprocess_image(img_path, out_path)

            records.append({
                "report_id": report_id,
                "source_page_path": str(img_path.resolve()),
                "processed_page_path": meta["processed_image_path"],
                "file_name": img_path.name,
                "estimated_skew_angle": meta["estimated_skew_angle"],
                "threshold_otsu": meta["threshold_otsu"],
                "width": meta["width"],
                "height": meta["height"],
            })

        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")

    if not records:
        print("No processed images created.")
        return

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "source_page_path",
                "processed_page_path",
                "file_name",
                "estimated_skew_angle",
                "threshold_otsu",
                "width",
                "height",
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Done. Processed images: {len(records)}")
    print(f"Errors: {errors}")
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()