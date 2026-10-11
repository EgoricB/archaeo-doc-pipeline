import argparse
import csv
import os
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path

# Размер страниц ограничен MAX_PAGE_PIXELS (см. page_dpis), поэтому защита Pillow
# от «декомпрессионных бомб» здесь не нужна и только выдаёт ложные предупреждения.
Image.MAX_IMAGE_PIXELS = None

from preprocess_pages import preprocess_image
from layout import analyze_layout
from run_ocr import run_tesseract
from linguistics import extract_entities, corpus_stats
from reporting import save_outputs
from utils import ensure_dir


from config import (POPPLER_PATH as POPLER_PATH, RENDER_DPI as DPI, BINARIZE, TESSDATA_DIR, OCR_LANG,
                    SAVE_RAW_PAGES, SAVE_LAYOUT_DEBUG, RENDER_CHUNK, WORKERS, MAX_PAGE_PIXELS)
FMT = "png"


def safe_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r'[\\/*?:"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name or "document"


def find_pdfs(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf"])


def page_dpis(pdf_path: Path, n_pages: int):
    """dpi для каждой страницы: RENDER_DPI, а для больших листов — меньше, чтобы не превысить MAX_PAGE_PIXELS."""
    info = pdfinfo_from_path(str(pdf_path), poppler_path=POPLER_PATH, first_page=1, last_page=n_pages)
    dpis = []
    for i in range(1, n_pages + 1):
        size = info.get(f"Page {i:>4} size") or (info.get("Page size") if i == 1 else None)
        m = re.match(r"\s*([\d.]+)\s*x\s*([\d.]+)", size or "")
        dpi = DPI
        if m:
            w_in, h_in = float(m.group(1)) / 72, float(m.group(2)) / 72
            if w_in * h_in * DPI * DPI > MAX_PAGE_PIXELS:
                dpi = max(50, int((MAX_PAGE_PIXELS / (w_in * h_in)) ** 0.5))
        dpis.append(dpi)
    return dpis


def iter_pdf_pages(pdf_path: Path, output_dir: Path):
    """Рендерит PDF порциями по RENDER_CHUNK страниц и отдаёт (строка манифеста, картинка).

    В памяти одновременно не больше RENDER_CHUNK страниц, поэтому размер отчёта
    не важен. Исходный рендер сохраняется на диск, только если SAVE_RAW_PAGES = True.
    """
    n_pages = int(pdfinfo_from_path(str(pdf_path), poppler_path=POPLER_PATH)["Pages"])
    dpis = page_dpis(pdf_path, n_pages)
    pages_raw_dir = ensure_dir(output_dir / "pages_raw") if SAVE_RAW_PAGES else None

    first = 1
    while first <= n_pages:
        # Порция — подряд идущие страницы с одинаковым dpi, не больше RENDER_CHUNK
        dpi = dpis[first - 1]
        last = first
        while last < n_pages and last - first + 1 < RENDER_CHUNK and dpis[last] == dpi:
            last += 1
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=FMT,
            first_page=first,
            last_page=last,
            poppler_path=POPLER_PATH,
        )
        for idx, img in enumerate(images, start=first):
            page_path = ""
            if pages_raw_dir:
                page_path = pages_raw_dir / f"page_{idx:04d}.{FMT}"
                img.save(page_path)
                page_path = str(page_path.resolve())
            yield {"page_id": idx, "source": str(pdf_path.resolve()), "image_path": page_path, "render_dpi": dpi}, img
        del images
        first = last + 1


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
    debug_dir = ensure_dir(output_dir / "layout_debug") if SAVE_LAYOUT_DEBUG else None
    ocr_dir = ensure_dir(output_dir / "ocr_text")

    manifest = []
    page_rows = []
    entity_rows = []
    texts = []

    for row, img in iter_pdf_pages(pdf_path, output_dir):
        manifest.append(row)
        page_id = row["page_id"]
        processed_path = processed_dir / f"page_{page_id:04d}.png"
        debug_path = debug_dir / f"page_{page_id:04d}.png" if debug_dir else None
        ocr_base = ocr_dir / f"page_{page_id:04d}"

        pre = preprocess_image(img, processed_path)
        img.close()
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
            "render_dpi": row["render_dpi"],
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
        "entity_rows": entity_rows,
    }


def init_worker():
    # Несколько процессов Tesseract с OpenMP-потоками мешают друг другу;
    # при параллельной обработке отчётов каждый Tesseract работает в один поток.
    os.environ["OMP_THREAD_LIMIT"] = "1"


def run_job(pdf_path: Path, doc_output_dir: Path):
    """Обработка одного отчёта в отдельном процессе. Исключения возвращаются, а не теряются."""
    t0 = time.time()
    try:
        result = process_one_pdf(pdf_path, doc_output_dir)
        result["status"] = "success"
    except Exception as e:
        result = {"status": "failed", "error": str(e), "traceback": traceback.format_exc(), "entity_rows": []}
    result["seconds"] = round(time.time() - t0, 1)
    return result


def load_entities(doc_output_dir: Path):
    """Сущности уже обработанного отчёта — чтобы общая частотная таблица была полной при --skip-existing."""
    path = doc_output_dir / "entities.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path, usecols=["type", "text"], dtype=str).fillna("").to_dict("records")
    except (ValueError, pd.errors.EmptyDataError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input_reports", help="Folder with PDF files")
    ap.add_argument("--output", default="results", help="Output root directory")
    ap.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have report.md")
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"Reports processed in parallel (default {WORKERS})")
    args = ap.parse_args()
    workers = max(1, args.workers)

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
    model = Path(TESSDATA_DIR) / f"{OCR_LANG}.traineddata"
    print(f"Render: {DPI} dpi, binarize: {BINARIZE}")
    print(f"OCR model: {model if model.exists() else 'installed with Tesseract (tessdata_best not found)'}")
    print(f"Max page size: {MAX_PAGE_PIXELS / 1e6:.0f} Mpx (larger sheets are rendered at lower dpi)")
    print(f"Workers: {workers}, render chunk: {RENDER_CHUNK} pages, "
          f"save raw pages: {SAVE_RAW_PAGES}, save layout debug: {SAVE_LAYOUT_DEBUG}")

    jobs = []
    for pdf_path in pdf_files:
        doc_output_dir = output_root / safe_name(pdf_path.name)
        if args.skip_existing and (doc_output_dir / "report.md").exists():
            ents = load_entities(doc_output_dir)
            all_entity_rows.extend(ents)
            try:
                pa = pd.read_csv(doc_output_dir / "page_analysis.csv", usecols=["ocr_available"])
                pages, ocr_pages = len(pa), int(pa["ocr_available"].astype(str).eq("True").sum())
            except (OSError, ValueError, pd.errors.EmptyDataError):
                pages, ocr_pages = "", ""
            summary_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "status": "skipped",
                "pages": pages,
                "ocr_pages": ocr_pages,
                "entities": len(ents),
                "output_dir": str(doc_output_dir.resolve()),
                "message": "report.md already exists",
            })
        else:
            jobs.append((pdf_path, doc_output_dir))

    print(f"Skipped (already processed): {len(summary_rows)}, to process: {len(jobs)}")
    # Крупные отчёты — первыми, чтобы в конце не ждать один большой файл
    jobs.sort(key=lambda j: j[0].stat().st_size, reverse=True)

    def record(pdf_path, doc_output_dir, result, done):
        if result["status"] == "success":
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
            print(f"[{done}/{len(jobs)}] OK {pdf_path.name}: pages={result['pages']}, "
                  f"entities={result['entities']}, {result['seconds']} s")
        else:
            summary_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "status": "failed",
                "pages": "",
                "ocr_pages": "",
                "entities": "",
                "output_dir": str(doc_output_dir.resolve()),
                "message": result["error"],
            })
            error_rows.append({
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path.resolve()),
                "error": result["error"],
                "traceback": result["traceback"],
            })
            print(f"[{done}/{len(jobs)}] ERROR {pdf_path.name}: {result['error']}")

    t_start = time.time()
    if workers == 1:
        for done, (pdf_path, doc_output_dir) in enumerate(jobs, start=1):
            print(f"[{done}/{len(jobs)}] PROCESSING: {pdf_path.name}")
            record(pdf_path, doc_output_dir, run_job(pdf_path, doc_output_dir), done)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(run_job, p, d): (p, d) for p, d in jobs}
            for done, fut in enumerate(as_completed(futures), start=1):
                pdf_path, doc_output_dir = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:  # процесс упал целиком, например из-за нехватки памяти
                    result = {"status": "failed", "error": f"worker crashed: {e}",
                              "traceback": traceback.format_exc()}
                record(pdf_path, doc_output_dir, result, done)
    print(f"Processing time: {round((time.time() - t_start) / 60, 1)} min")

    summary_rows.sort(key=lambda r: r["pdf_name"])
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