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
    experiments/ocr_tuning/summary.md   — таблица и проверка значимости для README / Pull Request
    experiments/ocr_tuning/work/        — картинки и тексты (в git не попадают)

Повторный запуск продолжает с места остановки: готовые тексты не пересчитываются.

Проверка значимости — парный бутстреп: страницы выборки много раз случайно
перевыбираются с возвращением, и для каждой перевыборки считается разница между
вариантами. Разница значима, если её 95-процентный интервал не включает ноль.
Пересчитать только статистику по готовому pages.csv, без OCR:
    python experiments/ocr_tuning/run_experiment.py --analyze-only
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


# ---------- проверка значимости ----------

N_BOOT = 5000
BOOT_SEED = 0


def paired_arrays(rows, variants):
    """Матрицы known/checked: строка — страница, столбец — вариант. Только страницы, есть во всех вариантах."""
    by = {}
    for r in rows:
        by.setdefault((r["document"], int(r["page_id"])), {})[r["variant"]] = r
    keys = [k for k, v in by.items() if all(name in v for name in variants)]
    known = np.array([[by[k][v]["known_words"] for v in variants] for k in keys], dtype=float)
    checked = np.array([[by[k][v]["checked_words"] for v in variants] for k in keys], dtype=float)
    return known, checked


def bootstrap(rows, variants, pairs):
    """Для каждой пары (a, b) — 95 % интервалы разницы долей словарных слов (п.п.)
    и относительной разницы числа правильных слов (%)."""
    known, checked = paired_arrays(rows, variants)
    n = len(known)
    idx = {v: i for i, v in enumerate(variants)}
    rng = np.random.default_rng(BOOT_SEED)
    samples = rng.integers(0, n, size=(N_BOOT, n))
    K = known[samples].sum(axis=1)      # N_BOOT × вариантов
    C = checked[samples].sum(axis=1)
    rate = np.divide(K, C, out=np.zeros_like(K), where=C > 0)

    out = []
    for a, b in pairs:
        ia, ib = idx[a], idx[b]
        d_rate = (rate[:, ia] - rate[:, ib]) * 100
        d_known = np.divide(K[:, ia], K[:, ib], out=np.ones(N_BOOT), where=K[:, ib] > 0) * 100 - 100
        point_rate = (known[:, ia].sum() / checked[:, ia].sum() - known[:, ib].sum() / checked[:, ib].sum()) * 100
        point_known = (known[:, ia].sum() / known[:, ib].sum() - 1) * 100
        lo_r, hi_r = np.percentile(d_rate, [2.5, 97.5])
        lo_k, hi_k = np.percentile(d_known, [2.5, 97.5])
        out.append({
            "a": a, "b": b, "pages": n,
            "rate_diff_pp": round(point_rate, 2), "rate_ci_low": round(lo_r, 2), "rate_ci_high": round(hi_r, 2),
            "rate_significant": bool(lo_r > 0 or hi_r < 0),
            "known_diff_pct": round(point_known, 1), "known_ci_low": round(lo_k, 1), "known_ci_high": round(hi_k, 1),
            "known_significant": bool(lo_k > 0 or hi_k < 0),
            "pages_better": int((known[:, ia] > known[:, ib]).sum()),
            "pages_worse": int((known[:, ia] < known[:, ib]).sum()),
        })
    return out


def comparison_pairs(variants, best):
    pairs = [(v, "baseline") for v in variants if v != "baseline" and "baseline" in variants]
    # Ключевые вопросы эксперимента: отдельный вклад каждого изменения
    steps = [("gray200", "baseline"), ("gray300", "gray200"), ("gray300_best", "gray300"),
             ("gray300_std", "gray300"), ("gray300_psm4", "gray300"), ("gray300_psm6", "gray300"),
             ("otsu300", "gray300"), ("sauvola300", "gray300")]
    pairs += [p for p in steps if p[0] in variants and p[1] in variants and p not in pairs]
    pairs += [(best, v) for v in variants if v != best and (best, v) not in pairs]
    return pairs


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


def fmt(x, digits=1):
    return f"{x:+.{digits}f}".replace(".", ",")


def sig_table(tests, title, intro):
    lines = ["", f"### {title}", "", intro, "",
             "| Сравнение | Δ словарных слов, п.п. [95 % ДИ] | Δ правильных слов, % [95 % ДИ] | Страниц лучше / хуже | Вывод |",
             "|---|---|---|---|---|"]
    for t in tests:
        if t["known_significant"] and t["rate_significant"]:
            verdict = "значимо лучше" if t["known_diff_pct"] > 0 else "значимо хуже"
        elif t["known_significant"]:
            verdict = ("больше правильных слов" if t["known_diff_pct"] > 0 else "меньше правильных слов") + ", доля — незначимо"
        elif t["rate_significant"]:
            verdict = ("доля выше" if t["rate_diff_pp"] > 0 else "доля ниже") + ", число слов — незначимо"
        else:
            verdict = "разница незначима"
        lines.append(
            f"| `{t['a']}` vs `{t['b']}` | {fmt(t['rate_diff_pp'])} [{fmt(t['rate_ci_low'])}; {fmt(t['rate_ci_high'])}] | "
            f"{fmt(t['known_diff_pct'])} [{fmt(t['known_ci_low'])}; {fmt(t['known_ci_high'])}] | "
            f"{t['pages_better']} / {t['pages_worse']} | {verdict} |"
        )
    return lines


def write_md(summary, path, n_pages, n_docs, tests=None, best_known=None):
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
    lines += ["", f"Наибольшая доля словарных слов: **`{best['variant']}`** — {best['description']}."]
    if best_known:
        lines += [f"Наибольшее число правильных слов: **`{best_known}`** — {VARIANTS[best_known][4]}."]

    if tests:
        lines += ["", "## Проверка значимости", "",
                  f"Парный бутстреп по страницам: {N_BOOT} перевыборок, 95 % доверительные интервалы (ДИ). "
                  "Сравниваются две величины: разница доли словарных слов в процентных пунктах "
                  "и относительная разница числа правильных (словарных) слов. "
                  "Разница значима, если интервал не включает ноль. "
                  "«Страниц лучше / хуже» — на скольких страницах первый вариант дал больше / меньше правильных слов."]
        groups = [
            ("Каждый вариант против исходного", "Помогает ли изменение по сравнению с текущим конвейером.",
             [t for t in tests if t["b"] == "baseline"]),
            ("Вклад отдельных изменений", "Каждое изменение сравнивается с ближайшим вариантом без него.",
             [t for t in tests if t["b"] != "baseline" and t["a"] != best_known]),
            (f"Лучший вариант против остальных", f"Отличается ли `{best_known}` от остальных вариантов.",
             [t for t in tests if t["a"] == best_known and t["b"] != "baseline"]),
        ]
        for title, intro, ts in groups:
            if ts:
                lines += sig_table(ts, title, intro)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows, variants, n_pages, n_docs):
    summary = summarize(rows, variants)
    best_known = max(summary, key=lambda s: sum(r["known_words"] for r in rows if r["variant"] == s["variant"]))["variant"]
    tests = bootstrap(rows, variants, comparison_pairs(variants, best_known)) if len(variants) > 1 else []
    write_csv(HERE / "pages.csv", rows)
    write_csv(HERE / "summary.csv", summary)
    if tests:
        write_csv(HERE / "significance.csv", tests)
    write_md(summary, HERE / "summary.md", n_pages, n_docs, tests, best_known)
    print("\n" + (HERE / "summary.md").read_text(encoding="utf-8"))


def load_pages_csv():
    path = HERE / "pages.csv"
    if not path.exists():
        raise SystemExit(f"Нет {path}: сначала запустите эксперимент без --analyze-only")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ("page_id", "chars", "words", "checked_words", "known_words"):
                r[k] = int(r[k])
            for k in ("confidence", "seconds", "lexicon_rate", "garbage_rate", "mixed_script_rate", "short_token_rate"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


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
    ap.add_argument("--analyze-only", action="store_true",
                    help="Только пересчитать сводку и значимость по готовому pages.csv")
    args = ap.parse_args()

    if args.analyze_only:
        rows = load_pages_csv()
        variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
        keys = {(r["document"], r["page_id"]) for r in rows}
        write_outputs(rows, variants, len(keys), len({k[0] for k in keys}))
        return

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

    write_outputs(rows, args.variants, len(pages), n_docs)


if __name__ == "__main__":
    main()
