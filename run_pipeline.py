import argparse
import csv
import re
import traceback
from collections import Counter
from pathlib import Path

from pdf2image import convert_from_path

from preprocess_pages import preprocess_image
from layout import analyze_layout
from run_ocr import run_tesseract
from linguistics import extract_entities, corpus_stats
from reporting import save_outputs
from utils import ensure_dir


from config import POPPLER_PATH as POPLER_PATH
DPI = 200
FMT = "png"


def safe_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name or "document"


def find_pdfs(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf"])


def render_pdf_to_manifest(pdf_path: Path, output_dir: Path):
    pages_raw_dir = ensure_dir(output_dir / "pages_raw")
    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        fmt=FMT,
        thread_count=2,
        poppler_path=POPLER_PATH,
    )

    manifest = []
    for idx, img in enumerate(pages, start=1):
        page_path = pages_raw_dir / f"page_{idx:04d}.{FMT}"
        img.save(page_path)
        manifest.append({
            "page_id": idx,
            "source": str(pdf_path.resolve()),
            "image_path": str(page_path.resolve()),
        })
    return manifest


def aggregate_entity_frequencies(all_entity_rows, output_csv: Path):
    counter = Counter()
    for row in all_entity_rows:
        key = (row.get("type", ""), row.get("text", ""))
        counter[key] += 1

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "text", "count"])
        writer.writeheader()
        for (entity_type, entity_text), count in counter.most_common():
            writer.writerow({
                "type": entity_type,
                "text": entity_text,
                "count": count,
            })


def process_one_pdf(pdf_path: Path, output_dir: Path):
    output_dir = ensure_dir(output_dir)
    processed_dir = ensure_dir(output_dir / "processed_pages")
    debug_dir = ensure_dir(output_dir / "layout_debug")
    ocr_dir = ensure_dir(output_dir / "ocr_text")

    manifest = render_pdf_to_manifest(pdf_path, output_dir)

    page_rows = []
    entity_rows = []
    texts = []

    for row in manifest:
        page_id = row["page_id"]
        image_path = Path(row["image_path"])
        processed_path = processed_dir / f"page_{page_id:04d}.png"
        debug_path = debug_dir / f"page_{page_id:04d}.png"
        ocr_base = ocr_dir / f"page_{page_id:04d}"

        pre = preprocess_image(image_path, processed_path)
        lay = analyze_layout(processed_path, debug_path)
        ocr = run_tesseract(processed_path, ocr_base)

        text = ocr.get("ocr_text", "") or ""
        texts.append(text)

        ents = extract_entities(text)
        for e in ents:
            entity_rows.append({
                "page_id": page_id,
                "source_pdf": str(pdf_path.resolve()),
                **e
            })

        page_rows.append({
            "page_id": page_id,
            "source": row["source"],
            "image_path": row["image_path"],
            "source_pdf": str(pdf_path.resolve()),
            **pre,
            **lay,
            "ocr_available": ocr.get("ocr_available", False),
            "ocr_text_path": ocr.get("text_path"),
            "ocr_error": ocr.get("ocr_error", ""),
            "num_entities": len(ents),
            "num_chars_ocr": len(text),
        })

    stats = corpus_stats(texts)
    save_outputs(output_dir, manifest, page_rows, entity_rows, stats)

    return {
        "pages": len(page_rows),
        "entities": len(entity_rows),
        "ocr_pages": sum(1 for r in page_rows if r.get("ocr_available")),
        "page_rows": page_rows,
        "entity_rows": entity_rows,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input_reports", help="Folder with PDF files")
    ap.add_argument("--output", default="results", help="Output root directory")
    ap.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have report.md")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_root = ensure_dir(args.output)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}")
        return

    poppler_dir = Path(POPLER_PATH)
    if not poppler_dir.exists():
        print(f"ERROR: Poppler folder not found: {POPLER_PATH}")
        return
    if not (poppler_dir / "pdftoppm.exe").exists():
        print(f"ERROR: pdftoppm.exe not found in: {POPLER_PATH}")
        return
    if not (poppler_dir / "pdfinfo.exe").exists():
        print(f"ERROR: pdfinfo.exe not found in: {POPLER_PATH}")
        return

    pdf_files = find_pdfs(input_dir)
    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return

    batch_summary_path = output_root / "batch_summary.csv"
    batch_errors_path = output_root / "batch_errors.csv"
    batch_entity_freq_path = output_root / "batch_entity_frequencies.csv"

    summary_rows = []
    error_rows = []
    all_entity_rows = []

    print(f"Found PDF files: {len(pdf_files)}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_root}")
    print(f"Using Poppler: {POPLER_PATH}")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        doc_name = safe_name(pdf_path.name)
        doc_output_dir = output_root / doc_name
        report_md = doc_output_dir / "report.md"

        if args.skip_existing and report_md.exists():
            print(f"[{idx}/{len(pdf_files)}] SKIP {pdf_path.name} -> already processed")
            summary_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "status": "skipped",
                "pages": "",
                "ocr_pages": "",
                "entities": "",
                "output_dir": str(doc_output_dir.resolve()),
                "message": "report.md already exists",
            })
            continue

        print(f"\n[{idx}/{len(pdf_files)}] PROCESSING: {pdf_path.name}")

        try:
            result = process_one_pdf(pdf_path, doc_output_dir)
            all_entity_rows.extend(result["entity_rows"])

            summary_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "status": "success",
                "pages": result["pages"],
                "ocr_pages": result["ocr_pages"],
                "entities": result["entities"],
                "output_dir": str(doc_output_dir.resolve()),
                "message": "",
            })

            print(
                f"  OK: pages={result['pages']}, "
                f"ocr_pages={result['ocr_pages']}, "
                f"entities={result['entities']}"
            )

        except Exception as e:
            tb = traceback.format_exc()

            summary_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "status": "failed",
                "pages": "",
                "ocr_pages": "",
                "entities": "",
                "output_dir": str(doc_output_dir.resolve()),
                "message": str(e),
            })

            error_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "error": str(e),
                "traceback": tb,
            })

            print(f"  ERROR: {e}")

    with open(batch_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_name",
                "pdf_path",
                "status",
                "pages",
                "ocr_pages",
                "entities",
                "output_dir",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(batch_errors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_name",
                "pdf_path",
                "error",
                "traceback",
            ],
        )
        writer.writeheader()
        writer.writerows(error_rows)

    if all_entity_rows:
        aggregate_entity_frequencies(all_entity_rows, batch_entity_freq_path)

    success_count = sum(1 for r in summary_rows if r["status"] == "success")
    failed_count = sum(1 for r in summary_rows if r["status"] == "failed")
    skipped_count = sum(1 for r in summary_rows if r["status"] == "skipped")

    print("\n=== BATCH COMPLETE ===")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Summary: {batch_summary_path}")
    print(f"Errors: {batch_errors_path}")
    if all_entity_rows:
        print(f"Entity frequencies: {batch_entity_freq_path}")


if __name__ == "__main__":
    main()