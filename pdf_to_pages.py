from pathlib import Path
import csv

from pdf2image import convert_from_path


INPUT_DIR = Path("input_reports")
PAGES_DIR = Path("pages")
MANIFEST_PATH = Path("pages_manifest.csv")

from config import POPPLER_PATH as POPLER_PATH

DPI = 200
FMT = "png"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_pdfs():
    if not INPUT_DIR.exists():
        print(f"Input dir not found: {INPUT_DIR}")
        return

    for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
        yield pdf_path


def report_id_from_name(pdf_path: Path) -> str:
    return pdf_path.stem


def report_output_dir(report_id: str) -> Path:
    return PAGES_DIR / report_id


def already_processed(report_id: str) -> bool:
    out_dir = report_output_dir(report_id)
    if not out_dir.exists():
        return False

    pngs = list(out_dir.glob("*.png"))
    jpgs = list(out_dir.glob("*.jpg"))
    jpegs = list(out_dir.glob("*.jpeg"))

    return len(pngs) + len(jpgs) + len(jpegs) > 0


def collect_existing_pages(report_id: str):
    out_dir = report_output_dir(report_id)

    page_files = sorted(
        list(out_dir.glob("*.png")) +
        list(out_dir.glob("*.jpg")) +
        list(out_dir.glob("*.jpeg"))
    )

    records = []
    for idx, page_path in enumerate(page_files, start=1):
        records.append({
            "report_id": report_id,
            "page_index": idx,
            "page_path": str(page_path.resolve()),
        })

    return records


def convert_pdf_to_pages(pdf_path: Path, report_id: str):
    out_dir = ensure_dir(report_output_dir(report_id))

    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        fmt=FMT,
        thread_count=2,
        poppler_path=POPLER_PATH,
    )

    records = []

    for idx, img in enumerate(pages, start=1):
        page_filename = f"{report_id}_page_{idx:03d}.{FMT}"
        page_path = out_dir / page_filename
        img.save(page_path)

        records.append({
            "report_id": report_id,
            "page_index": idx,
            "page_path": str(page_path.resolve()),
        })

    return records


def write_manifest(records):
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "page_index", "page_path"]
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    ensure_dir(PAGES_DIR)

    poppler_dir = Path(POPLER_PATH)
    pdftoppm_exe = poppler_dir / "pdftoppm.exe"
    pdfinfo_exe = poppler_dir / "pdfinfo.exe"

    if not poppler_dir.exists():
        print(f"ERROR: Poppler folder not found: {POPLER_PATH}")
        return

    if not pdftoppm_exe.exists():
        print(f"ERROR: pdftoppm.exe not found: {pdftoppm_exe}")
        return

    if not pdfinfo_exe.exists():
        print(f"ERROR: pdfinfo.exe not found: {pdfinfo_exe}")
        return

    print(f"Using Poppler: {POPLER_PATH}")
    print(f"Settings: dpi={DPI}, fmt={FMT}, thread_count=2")

    all_records = []
    total_pdfs = 0
    skipped = 0
    processed = 0

    pdfs = list(iter_pdfs() or [])
    if not pdfs:
        print("No PDF files found in input_reports/")
        return

    for pdf_path in pdfs:
        total_pdfs += 1
        report_id = report_id_from_name(pdf_path)

        if already_processed(report_id):
            existing_records = collect_existing_pages(report_id)
            all_records.extend(existing_records)
            skipped += 1
            print(f"SKIP {pdf_path.name} -> already has {len(existing_records)} pages")
            continue

        print(f"Processing {pdf_path.name} -> report_id={report_id}")

        try:
            records = convert_pdf_to_pages(pdf_path, report_id)
            all_records.extend(records)
            processed += 1
            print(f"  OK: {len(records)} pages")

            # записываем manifest после каждого успешно обработанного PDF
            write_manifest(all_records)

        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_records:
        print("No pages processed successfully, manifest not written.")
        return

    write_manifest(all_records)

    print(f"Done. Manifest written to: {MANIFEST_PATH}")
    print(f"Total PDFs: {total_pdfs}")
    print(f"Processed now: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Total pages in manifest: {len(all_records)}")


if __name__ == "__main__":
    main()