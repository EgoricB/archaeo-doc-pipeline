#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def compute_basic_shape_features(img: Image.Image):
    w, h = img.size
    aspect = w / h if h > 0 else 0.0
    return w, h, aspect


def estimate_border_frame_fraction(img: Image.Image, border_px: int = 20, thresh: int = 230) -> float:
    w, h = img.size
    arr = np.array(img)

    top = arr[0:border_px, :]
    bottom = arr[h - border_px:h, :]
    left = arr[:, 0:border_px]
    right = arr[:, w - border_px:w]

    border_region = np.concatenate(
        [top.reshape(-1), bottom.reshape(-1), left.reshape(-1), right.reshape(-1)]
    )

    dark = (border_region < thresh).sum()
    total = border_region.size
    return dark / total if total > 0 else 0.0


def estimate_fill_fraction(img: Image.Image, thresh: int = 230) -> float:
    arr = np.array(img)
    dark = (arr < thresh).sum()
    total = arr.size
    return dark / total if total > 0 else 0.0


def enrich_plans_dataset(plans_csv: Path, results_root: Path, output_csv: Path):
    """
    results_root: путь до папки results, например:
      results

    Путь к изображению собирается как:
      results_root / report_id_dir / "processed_pages" / file_name
    """
    df = pd.read_csv(plans_csv)
    df.columns = [c.lower() for c in df.columns]

    if "report_id" not in df.columns or "file_name" not in df.columns:
        raise ValueError("Нужны колонки report_id и file_name в plans_dataset.csv")

    widths, heights, aspects = [], [], []
    border_frame_fracs, fill_fracs, has_frame_flags = [], [], []
    resolved_paths = []

    for _, row in df.iterrows():
        report_id = str(row["report_id"])
        file_name = str(row["file_name"])

        # В твоей структуре: results/<report_id>/processed_pages/<file_name>
        img_path = results_root / report_id / "processed_pages" / file_name

        resolved_paths.append(str(img_path))

        if not img_path.exists():
            widths.append(np.nan)
            heights.append(np.nan)
            aspects.append(np.nan)
            border_frame_fracs.append(np.nan)
            fill_fracs.append(np.nan)
            has_frame_flags.append(False)
            continue

        img = load_image(img_path)
        w, h, aspect = compute_basic_shape_features(img)
        widths.append(w)
        heights.append(h)
        aspects.append(aspect)

        border_frac = estimate_border_frame_fraction(img)
        border_frame_fracs.append(border_frac)

        fill_frac = estimate_fill_fraction(img)
        fill_fracs.append(fill_frac)

        has_frame_flags.append(border_frac > 0.05)

    df["resolved_image_path"] = resolved_paths
    df["img_width"] = widths
    df["img_height"] = heights
    df["img_aspect_ratio"] = aspects
    df["border_frame_fraction"] = border_frame_fracs
    df["fill_fraction"] = fill_fracs
    df["has_frame"] = has_frame_flags

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved enriched plans dataset to: {output_csv}")
    print(df[["heuristic_label", "report_id", "file_name",
              "img_width", "img_height",
              "img_aspect_ratio", "border_frame_fraction",
              "fill_fraction", "has_frame"]].head(10))


def main():
    parser = argparse.ArgumentParser(description="Enrich plans_dataset.csv with basic CV-based features.")
    parser.add_argument("--plans-csv", type=str, required=True,
                        help="Path to plans_dataset.csv.")
    parser.add_argument("--results-root", type=str, required=True,
                        help="Root directory where 'results' with processed_pages live.")
    parser.add_argument("--output", type=str, default="plans_dataset_enriched.csv",
                        help="Output CSV path (default: plans_dataset_enriched.csv).")
    args = parser.parse_args()

    enrich_plans_dataset(
        plans_csv=Path(args.plans_csv),
        results_root=Path(args.results_root),
        output_csv=Path(args.output),
    )


if __name__ == "__main__":
    main()