from pathlib import Path
import csv
import re
from collections import defaultdict


INPUT_MANIFEST = Path("ocr_manifest.csv")
OUTPUT_DIR = Path("ocr_reports")
OUTPUT_MANIFEST = Path("ocr_reports_manifest.csv")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_page_number(file_name: str) -> int:
    m = re.search(r"_page_(\d+)", file_name)
    if not m:
        return 10**9
    return int(m.group(1))


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def safe_word_count(text: str) -> int:
    return len(text.split())


def russian_letter_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁё]", text))


def digit_count(text: str) -> int:
    return len(re.findall(r"\d", text))


def build_doc_quality_label(total_chars: int, total_words: int, rus_letters: int) -> str:
    if total_chars < 300 or total_words < 50:
        return "low_text"
    if rus_letters < max(50, total_chars * 0.15):
        return "noisy"
    return "usable"


def main():
    if not INPUT_MANIFEST.exists():
        print(f"ERROR: input manifest not found: {INPUT_MANIFEST}")
        return

    ensure_dir(OUTPUT_DIR)

    grouped = defaultdict(list)

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["report_id"]].append(row)

    if not grouped:
        print("No OCR rows found in manifest.")
        return

    output_records = []
    total_reports = 0

    for report_id, rows in grouped.items():
        total_reports += 1

        rows_sorted = sorted(rows, key=lambda r: extract_page_number(r["file_name"]))

        page_texts = []
        total_chars = 0
        total_words = 0
        total_rus_letters = 0
        total_digits = 0

        for row in rows_sorted:
            text_path = Path(row["ocr_text_path"])
            text = read_text(text_path).strip()

            page_texts.append(
                f"=== PAGE {extract_page_number(row['file_name'])} ===\n{text}\n"
            )

            total_chars += len(text)
            total_words += safe_word_count(text)
            total_rus_letters += russian_letter_count(text)
            total_digits += digit_count(text)

        full_text = "\n".join(page_texts).strip()

        out_path = OUTPUT_DIR / f"{report_id}.txt"
        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(full_text)

        quality_label = build_doc_quality_label(
            total_chars=total_chars,
            total_words=total_words,
            rus_letters=total_rus_letters,
        )

        output_records.append({
            "report_id": report_id,
            "report_text_path": str(out_path.resolve()),
            "page_count": len(rows_sorted),
            "char_count": total_chars,
            "word_count": total_words,
            "russian_letter_count": total_rus_letters,
            "digit_count": total_digits,
            "quality_label": quality_label,
        })

        print(f"[{total_reports}] Built report: {report_id} ({len(rows_sorted)} pages)")

    with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "report_text_path",
                "page_count",
                "char_count",
                "word_count",
                "russian_letter_count",
                "digit_count",
                "quality_label",
            ]
        )
        writer.writeheader()
        writer.writerows(output_records)

    print(f"Done. Reports built: {len(output_records)}")
    print(f"Manifest written to: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()