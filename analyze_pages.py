from pathlib import Path
import csv

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


INPUT_MANIFEST = Path("processed_pages_manifest.csv")
OUTPUT_CSV = Path("page_analysis.csv")


def classify_page(black_ratio, large_component_count, component_count):
    if black_ratio < 0.03 and large_component_count < 15:
        return "sparse"
    if black_ratio > 0.20:
        return "dense_graphics"
    if large_component_count > 80:
        return "text_heavy"
    if component_count > 30:
        return "mixed"
    return "unclear"


def analyze_binary_page(image_path: Path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    binary_black = arr < 128

    height, width = binary_black.shape
    total_pixels = width * height
    black_pixels = int(binary_black.sum())
    black_ratio = black_pixels / total_pixels if total_pixels else 0.0

    labeled = label(binary_black)
    props = regionprops(labeled)

    component_count = len(props)

    large_components = [
        p for p in props
        if p.area >= 20
    ]
    large_component_count = len(large_components)

    if large_components:
        avg_component_area = float(np.mean([p.area for p in large_components]))
        median_component_area = float(np.median([p.area for p in large_components]))
    else:
        avg_component_area = 0.0
        median_component_area = 0.0

    page_type = classify_page(
        black_ratio=black_ratio,
        large_component_count=large_component_count,
        component_count=component_count,
    )

    return {
        "width": width,
        "height": height,
        "black_pixels": black_pixels,
        "black_ratio": round(black_ratio, 6),
        "component_count": component_count,
        "large_component_count": large_component_count,
        "avg_component_area": round(avg_component_area, 3),
        "median_component_area": round(median_component_area, 3),
        "page_type": page_type,
    }


def main():
    if not INPUT_MANIFEST.exists():
        print(f"ERROR: input manifest not found: {INPUT_MANIFEST}")
        return

    records = []
    errors = 0
    total = 0

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1
            image_path = Path(row["processed_page_path"])
            report_id = row["report_id"]
            file_name = row["file_name"]

            print(f"[{total}] Analyzing: {image_path}")

            try:
                features = analyze_binary_page(image_path)

                records.append({
                    "report_id": report_id,
                    "file_name": file_name,
                    "processed_page_path": str(image_path.resolve()),
                    "width": features["width"],
                    "height": features["height"],
                    "black_pixels": features["black_pixels"],
                    "black_ratio": features["black_ratio"],
                    "component_count": features["component_count"],
                    "large_component_count": features["large_component_count"],
                    "avg_component_area": features["avg_component_area"],
                    "median_component_area": features["median_component_area"],
                    "page_type": features["page_type"],
                })

            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")

    if not records:
        print("No page analysis results created.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "file_name",
                "processed_page_path",
                "width",
                "height",
                "black_pixels",
                "black_ratio",
                "component_count",
                "large_component_count",
                "avg_component_area",
                "median_component_area",
                "page_type",
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Done. Analyzed pages: {len(records)}")
    print(f"Errors: {errors}")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()