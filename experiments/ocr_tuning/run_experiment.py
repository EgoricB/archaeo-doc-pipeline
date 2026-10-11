"""
Эксперимент: как предобработка, разрешение и модель Tesseract влияют на качество OCR.

Берётся случайная выборка страниц типа `dense_text` из уже обработанных отчётов.
Каждая страница распознаётся в нескольких вариантах (см. VARIANTS), для каждого
варианта считаются метрики из evaluate_ocr_quality.py и средняя уверенность Tesseract.

Запуск из корня проекта:
    python experiments/ocr_tuning/run_experiment.py
    python experiments/ocr_tuning/run_experiment.py --pages 20 --variants baseline gray300

Результаты:
    experiments/ocr_tuning/pages.csv    — метрики по каждой странице и варианту
    experiments/ocr_tuning/summary.csv  — сводка по вариантам
    experiments/ocr_tuning/summary.md   — таблица для README / Pull Request
    experiments/ocr_tuning/work/        — картинки и тексты (в git не попадают)

Повторный запуск продолжает с места остановки: готовые тексты не пересчитываются.
"""

import argparse
import csv
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.filters import threshold_otsu, threshold_sauvola

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from evaluate_ocr_quality import page_metrics  # noqa: E402
from preprocess_pages import estimate_skew_angle  # noqa: E402

try:
    import pymorphy3
except ImportError:
    raise SystemExit("Нужен pymorphy3: pip install pymorphy3")

from pdf2image import convert_from_path  # noqa: E402


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"

MODEL_URLS = {
    "standard": "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata",
    "best": "https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata",
}

# name: (dpi, предобработка, psm, модель, описание)
VARIANTS = {
    "baseline":      (200, "otsu",    "3", "system",   "Текущий конвейер: 200 dpi, Оцу, psm 3, установленная модель"),
    "gray200":       (200, "gray",    "3", "system",   "Без бинаризации"),
    "gray300":       (300, "gray",    "3", "system",   "Без бинаризации, 300 dpi"),
    "otsu300":       (300, "otsu",    "3", "system",   "Оцу, 300 dpi"),
    "sauvola300":    (300, "sauvola", "3", "system",   "Адаптивная бинаризация Саувола, 300 dpi"),
    "gray300_psm4":  (300, "gray",    "4", "system",   "Без бинаризации, 300 dpi, psm 4 (одна колонка)"),
    "gray300_psm6":  (300, "gray",    "6", "system",   "Без бинаризации, 300 dpi, psm 6 (единый блок)"),
    "gray300_std":   (300, "gray",    "3", "standard", "Без бинаризации, 300 dpi, модель tessdata"),
    "gray300_best":  (300, "gray",    "3", "best",     "Без бинаризации, 300 dpi, модель tessdata_best"),
}


# ---------- выборка ----------

def find_pdf(source_pdf):
    p = Path(source_pdf)
    if p.exists():
        return p
    alt = ROOT / "input_reports" / Path(source_pdf.replace("\\", "/")).name
    return alt if alt.exists() else None


def sample_pages(results, n, seed, max_per_doc):
    by_doc = {}
    for csv_path in sorted(results.glob("*/page_analysis.csv")):
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r.get("page_type") == "dense_text"]
        if not rows:
            continue
        pdf = find_pdf(rows[0]["source_pdf"])
        if pdf is None:
            continue
        by_doc[csv_path.parent.name] = [(csv_path.parent.name, pdf, int(r["page_id"])) for r in rows]

    rng = random.Random(seed)
    docs = list(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])

    picked = []
    for round_ in range(max_per_doc):
        for d in docs:
            if len(picked) >= n:
                return picked
            if round_ < len(by_doc[d]):
                picked.append(by_doc[d][round_])
    return picked


# ---------- изображения ----------

def poppler_path():
    p = Path(config.POPPLER_PATH)
    return str(p) if p.exists() else None


def render(doc, pdf, page_id, dpi):
    out = WORK / "img" / f"{doc}_p{page_id:04d}_{dpi}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img = convert_from_path(pdf, dpi=dpi, first_page=page_id, last_page=page_id,
                                grayscale=True, poppler_path=poppler_path())[0]
        img.save(out)
    return Image.open(out).convert("L")


def skew_angle(gray):
    small = gray.copy()
    small.thumbnail((1000, 1000))
    return estimate_skew_angle(small)


def preprocess(gray, mode, angle, dpi):
    g = gray.filter(ImageFilter.MedianFilter(size=3))
    g = g.rotate(angle, expand=True, fillcolor=255)
    if mode == "gray":
        return g
    arr = np.array(g)
    if mode == "otsu":
        bw = arr > threshold_otsu(arr)
    elif mode == "sauvola":
        window = int(25 * dpi / 200) | 1
        bw = arr > threshold_sauvola(arr, window_size=window, k=0.2)
    else:
        raise ValueError(mode)
    return Image.fromarray((bw * 255).astype(np.uint8))


# ---------- Tesseract ----------

def model_dir(model):
    if model == "system":
        return None
    d = WORK / f"tessdata_{model}"
    f = d / "rus.traineddata"
    if not f.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю модель {model}: {MODEL_URLS[model]}")
        urllib.request.urlretrieve(MODEL_URLS[model], f)
    return d


def run_tesseract(img_path, psm, tessdata):
    cmd = [str(config.TESSERACT_EXE), str(img_path), "stdout", "-l", "rus", "--psm", psm]
    if tessdata:
        cmd += ["--tessdata-dir", str(tessdata)]
    cmd += ["-c", "tessedit_create_tsv=1"]  # TSV без файла конфигурации: работает с любой папкой моделей
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", "ignore")[:500])
    return parse_tsv(res.stdout.decode("utf-8", "ignore"))


def parse_tsv(tsv):
    lines, confs = {}, []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        word = cols[11].strip()
        if not word:
            continue
        key = (int(cols[2]), int(cols[3]), int(cols[4]))  # block, par, line
        lines.setdefault(key, []).append(word)
        conf = float(cols[10])
        if conf >= 0:
            confs.append(conf)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text, (statistics.mean(confs) if confs else 0.0)


# ---------- отчёт ----------

def pct(x):
    return f"{x * 100:.1f}".replace(".", ",")


def summarize(rows, variants):
    out = []
    for name in variants:
        rs = [r for r in rows if r["variant"] == name]
        if not rs:
            continue
        checked = sum(r["checked_words"] for r in rs)
        known = sum(r["known_words"] for r in rs)
        out.append({
            "variant": name,
            "description": VARIANTS[name][4],
            "pages": len(rs),
            "lexicon_rate": round(known / checked, 4) if checked else 0.0,
            "median_page_lexicon": round(statistics.median(r["lexicon_rate"] for r in rs), 4),
            "good_or_fair_share": round(sum(r["quality"] in ("good", "fair") for r in rs) / len(rs), 4),
            "empty_share": round(sum(r["quality"] == "empty" for r in rs) / len(rs), 4),
            "mean_confidence": round(statistics.mean(r["confidence"] for r in rs), 1),
            "sec_per_page": round(statistics.mean(r["seconds"] for r in rs), 2),
        })
    return out


def write_md(summary, path, n_pages, n_docs):
    base = next((s for s in summary if s["variant"] == "baseline"), None)
    lines = [
        "# Эксперимент: настройки OCR",
        "",
        f"Выборка: {n_pages} страниц типа `dense_text` из {n_docs} отчётов. "
        "Метрика — доля распознанных слов, которые есть в словаре (pymorphy3), "
        "см. `evaluate_ocr_quality.py`.",
        "",
        "| Вариант | Что изменено | Словарных слов, % | Δ к базовому | Хороших и средних страниц, % | Пустых, % | Уверенность Tesseract | сек/стр |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summary, key=lambda s: (s["variant"] != "baseline", -s["lexicon_rate"])):
        delta = "—" if base is None or s is base else f"{(s['lexicon_rate'] - base['lexicon_rate']) * 100:+.1f}".replace(".", ",")
        lines.append(
            f"| `{s['variant']}` | {s['description']} | {pct(s['lexicon_rate'])} | {delta} | "
            f"{pct(s['good_or_fair_share'])} | {pct(s['empty_share'])} | "
            f"{s['mean_confidence']:.0f} | {s['sec_per_page']:.1f} |".replace(".", ",")
        )
    best = max(summary, key=lambda s: s["lexicon_rate"])
    lines += ["", f"Лучший вариант: **`{best['variant']}`** — {best['description']}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Эксперимент с настройками OCR")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--pages", type=int, default=50)
    ap.add_argument("--max-per-doc", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()

    if not Path(config.TESSERACT_EXE).exists():
        raise SystemExit(f"Не найден Tesseract: {config.TESSERACT_EXE}")

    pages = sample_pages(Path(args.results), args.pages, args.seed, args.max_per_doc)
    if not pages:
        raise SystemExit("Не нашлось страниц dense_text с доступными PDF")
    n_docs = len({p[0] for p in pages})
    print(f"Выборка: {len(pages)} страниц из {n_docs} отчётов, вариантов: {len(args.variants)}")

    morph = pymorphy3.MorphAnalyzer()
    cache = {}
    angles = {}
    rows = []
    total = len(pages) * len(args.variants)
    step = 0

    for name in args.variants:
        dpi, mode, psm, model, _ = VARIANTS[name]
        tessdata = model_dir(model)
        for doc, pdf, page_id in pages:
            step += 1
            txt_path = WORK / "txt" / name / f"{doc}_p{page_id:04d}.txt"
            meta_path = txt_path.with_suffix(".meta")
            t0 = time.time()
            if txt_path.exists() and meta_path.exists():
                text = txt_path.read_text(encoding="utf-8")
                conf, secs = map(float, meta_path.read_text().split())
            else:
                gray = render(doc, pdf, page_id, dpi)
                key = (doc, page_id)
                if key not in angles:
                    angles[key] = skew_angle(render(doc, pdf, page_id, 200))
                img = preprocess(gray, mode, angles[key], dpi)
                img_path = WORK / "prep" / name / f"{doc}_p{page_id:04d}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)
                text, conf = run_tesseract(img_path, psm, tessdata)
                secs = time.time() - t0
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(text, encoding="utf-8")
                meta_path.write_text(f"{conf} {secs}")
            m = page_metrics(text, morph, cache)
            rows.append({"variant": name, "document": doc, "page_id": page_id,
                         "confidence": round(conf, 1), "seconds": round(secs, 2), **m})
            print(f"[{step}/{total}] {name:13s} {doc} p{page_id}: "
                  f"словарных {m['lexicon_rate']:.2f}, уверенность {conf:.0f}")

    summary = summarize(rows, args.variants)
    write_csv(HERE / "pages.csv", rows)
    write_csv(HERE / "summary.csv", summary)
    write_md(summary, HERE / "summary.md", len(pages), n_docs)
    print("\n" + (HERE / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
