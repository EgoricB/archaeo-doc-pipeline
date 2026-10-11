#!/usr/bin/env python3
"""
build_plans_dataset.py

Формирует plans_dataset.csv для дальнейшего CV/ML-анализа планов, схем, разрезов и карт.

Ожидается, что у тебя уже есть (как в текущем проекте):
- page_analysis.csv:
    report_id,file_name,processed_page_path,width,height,black_pixels,black_ratio,
    component_count,large_component_count,avg_component_area,median_component_area,page_type

- ocr_manifest.csv:
    report_id,file_name,processed_page_path,ocr_text_path,char_count,word_count
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Any

import pandas as pd


PLAN_KEYWORDS = [
    "план", "разрез", "секция", "профиль", "схема", "карта",
    "рис.", "рисунок", "figure", "plan", "section", "map"
]


def ensure_column(df: pd.DataFrame, candidates, default=None):
    """
    Возвращает имя существующей колонки из candidates.
    Если ни одной нет — создаёт первую с default.
    """
    for c in candidates:
        if c in df.columns:
            return c
    name = candidates[0]
    df[name] = default
    return name


def load_page_analysis(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"page_analysis file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_ocr_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ocr_manifest file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def read_ocr_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_plan_keyword(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in PLAN_KEYWORDS)


def build_plans_dataset(
    page_analysis_path: Path,
    ocr_manifest_path: Path,
    pages_root: Path,
    output_path: Path,
    text_density_threshold: float = 0.3,
    low_text_chars_threshold: int = 300
):
    # 1. Загружаем данные
    page_df = load_page_analysis(page_analysis_path)
    ocr_df = load_ocr_manifest(ocr_manifest_path)

    # 2. Приводим ключевые колонки к общему виду
    # report_id и file_name есть в обоих файлах
    if "report_id" not in page_df.columns or "file_name" not in page_df.columns:
        raise ValueError("page_analysis.csv должен содержать колонки report_id и file_name")

    if "report_id" not in ocr_df.columns or "file_name" not in ocr_df.columns:
        raise ValueError("ocr_manifest.csv должен содержать колонки report_id и file_name")

    report_id_col = "report_id"
    file_name_col = "file_name"

    # page_type
    page_type_col = ensure_column(page_df, ["page_type"])

    # num_chars_ocr — берём из char_count
    num_chars_col = ensure_column(ocr_df, ["char_count"], default=0)

    # text_density — у тебя её нет, поэтому используем black_ratio как прокси
    text_density_col = ensure_column(page_df, ["text_density"], default=None)
    if text_density_col not in page_df.columns or page_df[text_density_col].isna().all():
        # используем black_ratio как суррогат плотности заливки/чёрных пикселей
        if "black_ratio" in page_df.columns:
            text_density_col = "black_ratio"
        else:
            page_df["black_ratio"] = 0.0
            text_density_col = "black_ratio"

    # processed_page_path → путь к изображению
    image_path_col = ensure_column(page_df, ["processed_page_path"])

    # 3. Соединяем по report_id + file_name
    merged = pd.merge(
        page_df,
        ocr_df[[report_id_col, file_name_col, num_chars_col, "processed_page_path", "ocr_text_path"]],
        on=[report_id_col, file_name_col],
        how="left",
        suffixes=("", "_ocr")
    )

    # 4. Читаем OCR-текст и считаем has_plan_keyword
    #    Это может быть не очень быстро, но для 404 страниц нормально.
    def get_ocr_text(row) -> str:
        p = row.get("ocr_text_path")
        if not isinstance(p, str) or not p:
            return ""
        return read_ocr_text(Path(p))

    merged["ocr_text"] = merged.apply(get_ocr_text, axis=1)
    merged["has_plan_keyword"] = merged["ocr_text"].apply(detect_plan_keyword)

    # 5. Эвристический label
    def label_row(row: Dict[str, Any]) -> str:
        page_type = str(row[page_type_col]).lower()
        text_density = float(row[text_density_col]) if row[text_density_col] is not None else 0.0
        num_chars = int(row[num_chars_col]) if row[num_chars_col] is not None else 0
        has_kw = bool(row["has_plan_keyword"])

        # визуальные страницы: image_or_blank, фигуры, или низкая "плотность"
        is_visual = page_type in {"image_or_blank", "image", "figure", "photo"} or text_density < text_density_threshold
        low_text = num_chars < low_text_chars_threshold

        if is_visual and (has_kw or low_text):
            return "plan_candidate"
        if has_kw and not is_visual:
            return "maybe_plan"
        return "non_plan"

    merged["heuristic_label"] = merged.apply(label_row, axis=1)

    # 6. Формируем путь к изображению
    def build_image_path(row) -> str:
        # processed_page_path уже абсолютный путь; сделаем его относительным к pages_root, если возможно
        raw = row.get(image_path_col)
        if not isinstance(raw, str) or not raw:
            return ""
        p = Path(raw)
        try:
            return str(p.relative_to(pages_root))
        except Exception:
            # если не получается сделать относительным, оставим как есть
            return str(p)

    merged["page_image_path"] = merged.apply(build_image_path, axis=1)

    # 7. Собираем выходной датафрейм
    out_cols = [
        report_id_col,
        file_name_col,
        page_type_col,
        num_chars_col,
        text_density_col,
        "has_plan_keyword",
        "heuristic_label",
        "page_image_path",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]

    out_df = merged[out_cols].copy()

    # 8. Сохраняем CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved plans_dataset to: {output_path}")
    print(f"Total rows: {len(out_df)}")
    print("Label counts:")
    print(out_df["heuristic_label"].value_counts())


def main():
    parser = argparse.ArgumentParser(description="Build plans_dataset.csv from page and OCR manifests.")
    parser.add_argument("--page-analysis", type=str, required=True,
                        help="Path to page_analysis.csv (page-level metadata).")
    parser.add_argument("--ocr-manifest", type=str, required=True,
                        help="Path to ocr_manifest.csv (page-level OCR paths and counts).")
    parser.add_argument("--pages-root", type=str, required=True,
                        help="Root directory where page images are stored (used to relativize page_image_path).")
    parser.add_argument("--output", type=str, default="plans_dataset.csv",
                        help="Output CSV path (default: plans_dataset.csv in current dir).")
    parser.add_argument("--text-density-threshold", type=float, default=0.3,
                        help="Threshold for considering a page visually dominant (default: 0.3).")
    parser.add_argument("--low-text-chars-threshold", type=int, default=300,
                        help="Threshold for low OCR character count (default: 300).")
    args = parser.parse_args()

    build_plans_dataset(
        page_analysis_path=Path(args.page_analysis),
        ocr_manifest_path=Path(args.ocr_manifest),
        pages_root=Path(args.pages_root),
        output_path=Path(args.output),
        text_density_threshold=args.text_density_threshold,
        low_text_chars_threshold=args.low_text_chars_threshold,
    )


if __name__ == "__main__":
    main()