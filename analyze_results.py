from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")


def classify_document(row):
    if row["image_or_blank_share"] >= 0.50:
        return "visual_low_text"
    elif row["mixed_share"] >= 0.25:
        return "mixed_layout"
    elif row["dense_text_share"] >= 0.60 and row["image_or_blank_share"] <= 0.20:
        return "textual"
    else:
        return "hybrid_other"


def safe_mean(df, col):
    return df[col].mean() if col in df.columns else 0.0


def safe_median(df, col):
    return df[col].median() if col in df.columns else 0.0


def safe_sum(df, col):
    return df[col].sum() if col in df.columns else 0


def safe_count_by_type(df, page_type_value):
    if "page_type" not in df.columns:
        return 0
    return int((df["page_type"] == page_type_value).sum())


def main():
    if not RESULTS_DIR.exists():
        print(f"Папка не найдена: {RESULTS_DIR}")
        return

    doc_rows = []
    page_frames = []
    errors = []

    for csv_path in RESULTS_DIR.rglob("page_analysis.csv"):
        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                continue

            doc_name = csv_path.parent.name
            total_pages = len(df)

            dense_text_pages = safe_count_by_type(df, "dense_text")
            figure_or_table_pages = safe_count_by_type(df, "figure_or_table")
            mixed_pages = safe_count_by_type(df, "mixed")
            image_or_blank_pages = safe_count_by_type(df, "image_or_blank")

            avg_text_density = safe_mean(df, "text_density")
            median_text_density = safe_median(df, "text_density")

            total_entities = safe_sum(df, "num_entities")
            total_chars_ocr = safe_sum(df, "num_chars_ocr")
            avg_chars_ocr = safe_mean(df, "num_chars_ocr")
            median_chars_ocr = safe_median(df, "num_chars_ocr")

            low_ocr_pages_lt50 = int((df["num_chars_ocr"] < 50).sum()) if "num_chars_ocr" in df.columns else 0
            low_ocr_pages_lt200 = int((df["num_chars_ocr"] < 200).sum()) if "num_chars_ocr" in df.columns else 0
            low_ocr_pages_lt500 = int((df["num_chars_ocr"] < 500).sum()) if "num_chars_ocr" in df.columns else 0

            dense_text_share = dense_text_pages / total_pages if total_pages else 0.0
            figure_or_table_share = figure_or_table_pages / total_pages if total_pages else 0.0
            mixed_share = mixed_pages / total_pages if total_pages else 0.0
            image_or_blank_share = image_or_blank_pages / total_pages if total_pages else 0.0

            doc_rows.append({
                "document": doc_name,
                "pages": total_pages,
                "dense_text_pages": dense_text_pages,
                "figure_or_table_pages": figure_or_table_pages,
                "mixed_pages": mixed_pages,
                "image_or_blank_pages": image_or_blank_pages,
                "dense_text_share": round(dense_text_share, 4),
                "figure_or_table_share": round(figure_or_table_share, 4),
                "mixed_share": round(mixed_share, 4),
                "image_or_blank_share": round(image_or_blank_share, 4),
                "avg_text_density": round(avg_text_density, 6),
                "median_text_density": round(median_text_density, 6),
                "total_entities": int(total_entities),
                "total_chars_ocr": int(total_chars_ocr),
                "avg_chars_ocr": round(avg_chars_ocr, 2),
                "median_chars_ocr": round(median_chars_ocr, 2),
                "low_ocr_pages_lt50": low_ocr_pages_lt50,
                "low_ocr_pages_lt200": low_ocr_pages_lt200,
                "low_ocr_pages_lt500": low_ocr_pages_lt500,
                "source_csv": str(csv_path),
            })

            df = df.copy()
            df["document"] = doc_name
            df["source_csv"] = str(csv_path)
            page_frames.append(df)

        except Exception as e:
            errors.append((str(csv_path), str(e)))

    if not doc_rows:
        print("Не найдено ни одного валидного page_analysis.csv")
        return

    doc_df = pd.DataFrame(doc_rows)
    all_pages_df = pd.concat(page_frames, ignore_index=True)

    # Классы документов
    doc_df["document_class"] = doc_df.apply(classify_document, axis=1)

    # Общая сводка
    print("=== Общая сводка ===")
    print(f"Документов: {len(doc_df)}")
    print(f"Всего страниц: {len(all_pages_df)}")

    if "page_type" in all_pages_df.columns:
        print("\n=== Распределение типов страниц ===")
        print(all_pages_df["page_type"].value_counts())

        print("\n=== Доли типов страниц (%) ===")
        print((all_pages_df["page_type"].value_counts(normalize=True) * 100).round(2))

    if "num_chars_ocr" in all_pages_df.columns:
        print("\n=== OCR-статистика ===")
        print(f"Среднее число OCR-символов на страницу: {all_pages_df['num_chars_ocr'].mean():.2f}")
        print(f"Медиана OCR-символов на страницу: {all_pages_df['num_chars_ocr'].median():.2f}")
        print(f"Страниц с num_chars_ocr < 50: {(all_pages_df['num_chars_ocr'] < 50).sum()}")

    if "text_density" in all_pages_df.columns:
        print("\n=== Плотность текста ===")
        print(f"Средняя text_density: {all_pages_df['text_density'].mean():.6f}")
        print(f"Медиана text_density: {all_pages_df['text_density'].median():.6f}")

    print("\n=== Число документов по классам ===")
    print(doc_df["document_class"].value_counts())

    print("\n=== Число страниц по классам документов ===")
    print(doc_df.groupby("document_class")["pages"].sum().sort_values(ascending=False))

    print("\n=== Топ-10 документов по числу страниц ===")
    print(
        doc_df.sort_values("pages", ascending=False)
        .head(10)[["document", "pages", "document_class"]]
    )

    print("\n=== Топ-10 документов по доле image_or_blank ===")
    print(
        doc_df.sort_values("image_or_blank_share", ascending=False)
        .head(10)[
            ["document", "pages", "image_or_blank_pages", "image_or_blank_share", "document_class"]
        ]
    )

    print("\n=== Топ-10 документов по доле mixed ===")
    print(
        doc_df.sort_values("mixed_share", ascending=False)
        .head(10)[
            ["document", "pages", "mixed_pages", "mixed_share", "document_class"]
        ]
    )

    # 1. OCR по классам документов
    pages_with_class = all_pages_df.merge(
        doc_df[["document", "document_class"]],
        on="document",
        how="left",
        validate="many_to_one",
    )

    ocr_by_class = pages_with_class.groupby("document_class").agg(
        pages=("page_id", "count"),
        avg_chars_ocr=("num_chars_ocr", "mean"),
        median_chars_ocr=("num_chars_ocr", "median"),
        low_ocr_lt50=("num_chars_ocr", lambda s: (s < 50).sum()),
        low_ocr_share_lt50=("num_chars_ocr", lambda s: (s < 50).mean()),
        low_ocr_lt200=("num_chars_ocr", lambda s: (s < 200).sum()),
        low_ocr_share_lt200=("num_chars_ocr", lambda s: (s < 200).mean()),
        low_ocr_lt500=("num_chars_ocr", lambda s: (s < 500).sum()),
        low_ocr_share_lt500=("num_chars_ocr", lambda s: (s < 500).mean()),
        avg_text_density=("text_density", "mean"),
        median_text_density=("text_density", "median"),
    ).round(3)

    print("\n=== 1. OCR по классам документов ===")
    print(ocr_by_class)

    # 2. OCR по типам страниц
    ocr_by_page_type = all_pages_df.groupby("page_type").agg(
        pages=("page_id", "count"),
        avg_chars_ocr=("num_chars_ocr", "mean"),
        median_chars_ocr=("num_chars_ocr", "median"),
        low_ocr_lt50=("num_chars_ocr", lambda s: (s < 50).sum()),
        low_ocr_share_lt50=("num_chars_ocr", lambda s: (s < 50).mean()),
        low_ocr_lt200=("num_chars_ocr", lambda s: (s < 200).sum()),
        low_ocr_share_lt200=("num_chars_ocr", lambda s: (s < 200).mean()),
        low_ocr_lt500=("num_chars_ocr", lambda s: (s < 500).sum()),
        low_ocr_share_lt500=("num_chars_ocr", lambda s: (s < 500).mean()),
        avg_text_density=("text_density", "mean"),
        median_text_density=("text_density", "median"),
    ).round(3)

    print("\n=== 2. OCR по типам страниц ===")
    print(ocr_by_page_type)

    # 3. Layout-типология документов
    layout_map = doc_df[
        [
            "document",
            "document_class",
            "pages",
            "dense_text_pages",
            "figure_or_table_pages",
            "mixed_pages",
            "image_or_blank_pages",
            "dense_text_share",
            "figure_or_table_share",
            "mixed_share",
            "image_or_blank_share",
        ]
    ].copy()

    print("\n=== 3. Layout-карта документов ===")
    print(
        layout_map.sort_values(
            ["figure_or_table_share", "mixed_share", "image_or_blank_share"],
            ascending=False
        ).head(15)
    )

    print("\n=== Примеры документов по классам ===")
    for cls in ["visual_low_text", "mixed_layout", "textual", "hybrid_other"]:
        subset = doc_df[doc_df["document_class"] == cls]
        print(f"\n{cls}:")
        if subset.empty:
            print("Нет документов")
        else:
            print(
                subset.sort_values("pages", ascending=False)
                .head(10)[
                    [
                        "document",
                        "pages",
                        "dense_text_share",
                        "figure_or_table_share",
                        "mixed_share",
                        "image_or_blank_share",
                    ]
                ]
            )

    # Сохранение файлов
    doc_df.to_csv("results_document_summary_with_classes.csv", index=False, encoding="utf-8-sig")
    all_pages_df.to_csv("results_all_pages_merged.csv", index=False, encoding="utf-8-sig")
    ocr_by_class.to_csv("results_ocr_by_document_class.csv", encoding="utf-8-sig")
    ocr_by_page_type.to_csv("results_ocr_by_page_type.csv", encoding="utf-8-sig")
    layout_map.to_csv("results_document_layout_map.csv", index=False, encoding="utf-8-sig")

    print("\nСохранено:")
    print("- results_document_summary_with_classes.csv")
    print("- results_all_pages_merged.csv")
    print("- results_ocr_by_document_class.csv")
    print("- results_ocr_by_page_type.csv")
    print("- results_document_layout_map.csv")

    if errors:
        print("\n=== Ошибки чтения ===")
        for path, err in errors:
            print(f"{path}: {err}")


if __name__ == "__main__":
    main()