from pathlib import Path
import csv
import subprocess


INPUT_MANIFEST = Path("processed_pages_manifest.csv")
OCR_TEXT_DIR = Path("ocr_text")
OCR_MANIFEST_PATH = Path("ocr_manifest.csv")

from config import TESSERACT_EXE, TESSDATA_DIR, OCR_LANG, OCR_PSM as PSM


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_tesseract(image_path: Path, out_base: Path, lang: str = OCR_LANG, psm: str = PSM):
    if not TESSERACT_EXE.exists():
        return {
            "ocr_available": False,
            "text_path": None,
            "ocr_text": "",
            "ocr_error": f"tesseract.exe not found: {TESSERACT_EXE}",
        }

    cmd = [
        str(TESSERACT_EXE),
        str(image_path),
        str(out_base),
        "-l",
        lang,
        "--psm",
        psm,
    ]
    if (Path(TESSDATA_DIR) / f"{lang}.traineddata").exists():
        cmd += ["--tessdata-dir", str(TESSDATA_DIR)]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    txt_path = Path(str(out_base) + ".txt")

    if res.returncode != 0:
        return {
            "ocr_available": False,
            "text_path": None,
            "ocr_text": "",
            "ocr_error": res.stderr,
        }

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    return {
        "ocr_available": True,
        "text_path": str(txt_path.resolve()),
        "ocr_text": text,
        "ocr_error": "",
    }


def main():
    if not INPUT_MANIFEST.exists():
        print(f"ERROR: input manifest not found: {INPUT_MANIFEST}")
        return

    if not TESSERACT_EXE.exists():
        print(f"ERROR: tesseract.exe not found: {TESSERACT_EXE}")
        return

    ensure_dir(OCR_TEXT_DIR)

    print(f"Using Tesseract: {TESSERACT_EXE}")
    print(f"Settings: lang={OCR_LANG}, psm={PSM}")

    records = []
    total = 0
    errors = 0

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += 1

            report_id = row["report_id"]
            image_path = Path(row["processed_page_path"])
            file_name = row["file_name"]

            report_out_dir = OCR_TEXT_DIR / report_id
            ensure_dir(report_out_dir)

            out_base = report_out_dir / image_path.stem

            print(f"[{total}] OCR: {image_path}")

            result = run_tesseract(image_path, out_base)

            if not result["ocr_available"]:
                errors += 1
                print(f"  ERROR: {result.get('ocr_error', 'OCR failed')}")
                continue

            text = result["ocr_text"]

            records.append({
                "report_id": report_id,
                "file_name": file_name,
                "processed_page_path": str(image_path.resolve()),
                "ocr_text_path": result["text_path"],
                "char_count": len(text),
                "word_count": len(text.split()),
            })

    if not records:
        print("No OCR results created.")
        return

    with open(OCR_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "file_name",
                "processed_page_path",
                "ocr_text_path",
                "char_count",
                "word_count",
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Done. OCR pages: {len(records)}")
    print(f"Errors: {errors}")
    print(f"Manifest written to: {OCR_MANIFEST_PATH}")


if __name__ == "__main__":
    main()