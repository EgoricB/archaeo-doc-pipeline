"""
evaluate_ocr_quality.py

Оценка качества OCR без эталонной разметки.

Для каждой распознанной страницы считаются косвенные метрики:
- lexicon_rate      — доля русских слов (от 3 букв), которые есть в словаре
                      морфологического анализатора pymorphy3. Главная метрика:
                      приближённо показывает долю правильно распознанных слов;
- garbage_rate      — доля «мусорных» символов (не буквы, не цифры,
                      не обычная пунктуация) среди всех непробельных символов;
- mixed_script_rate — доля слов, в которых смешаны латиница и кириллица
                      (типичная ошибка OCR: «oбъект» с латинской «o»);
- short_token_rate  — доля однобуквенных фрагментов (признак разорванного текста).

Метка качества страницы:
- empty — меньше 20 слов (иллюстрация, пустая страница или OCR ничего не нашёл);
- good  — lexicon_rate >= 0.85;
- fair  — 0.70 <= lexicon_rate < 0.85;
- poor  — lexicon_rate < 0.70.

Запуск:
    pip install pymorphy3
    python evaluate_ocr_quality.py --results results --report docs/ocr_quality.md

Результаты:
    results/ocr_quality_pages.csv      — метрики по страницам
    results/ocr_quality_documents.csv  — метрики по документам
    docs/ocr_quality.md                — сводный отчёт
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pymorphy3
except ImportError:  # pragma: no cover
    raise SystemExit("Нужен pymorphy3: pip install pymorphy3")


MIN_WORDS = 20
GOOD, FAIR = 0.85, 0.70

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")
CYR_RE = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")
# Первые части сложных слов, которых нет в словаре как отдельных слов
COMPOUND_PARTS = {"юго", "северо", "северо", "восточно", "западно", "южно",
                  "ново", "старо", "древне", "ранне", "поздне", "средне", "верхне", "нижне"}
HAS_CYR = re.compile(r"[А-Яа-яЁё]")
HAS_LAT = re.compile(r"[A-Za-z]")
NORMAL_CHARS = re.compile(r"[A-Za-zА-Яа-яЁё0-9.,;:!?()\[\]\"'«»\-–—№%/+=°§*]")
HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")


def is_known(word, morph):
    if morph.word_is_known(word):
        return True
    if "-" in word:
        parts = word.split("-")
        return all(pt in COMPOUND_PARTS or morph.word_is_known(pt) for pt in parts)
    return False


def label_page(words, lexicon_rate):
    if words < MIN_WORDS:
        return "empty"
    if lexicon_rate >= GOOD:
        return "good"
    if lexicon_rate >= FAIR:
        return "fair"
    return "poor"


def page_metrics(text, morph, cache):
    text = HYPHEN_BREAK.sub(r"\1\2", text)  # склеиваем переносы «про-\nдолжение»
    non_space = [c for c in text if not c.isspace()]
    garbage = sum(1 for c in non_space if not NORMAL_CHARS.match(c))

    tokens = WORD_RE.findall(text)
    mixed = sum(1 for t in tokens if HAS_CYR.search(t) and HAS_LAT.search(t))
    short = sum(1 for t in tokens if len(t) == 1)

    checked = known = 0
    for t in tokens:
        if len(t) < 3 or not CYR_RE.match(t):
            continue
        w = t.lower()
        if w not in cache:
            cache[w] = is_known(w, morph)
        checked += 1
        known += cache[w]

    lexicon_rate = known / checked if checked else 0.0
    return {
        "chars": len(non_space),
        "words": len(tokens),
        "checked_words": checked,
        "known_words": known,
        "lexicon_rate": round(lexicon_rate, 4),
        "garbage_rate": round(garbage / len(non_space), 4) if non_space else 0.0,
        "mixed_script_rate": round(mixed / len(tokens), 4) if tokens else 0.0,
        "short_token_rate": round(short / len(tokens), 4) if tokens else 0.0,
        "quality": label_page(len(tokens), lexicon_rate),
    }


def load_page_types(doc_dir):
    path = doc_dir / "page_analysis.csv"
    types = {}
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                types[int(row["page_id"])] = row.get("page_type", "")
    return types


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def num(n):
    return f"{n:,}".replace(",", " ")


def pct(x):
    return f"{x * 100:.1f} %".replace(".", ",")


def build_report(pages, docs, worst_words):
    by_q = Counter(p["quality"] for p in pages)
    text_pages = [p for p in pages if p["quality"] != "empty"]
    checked = sum(p["checked_words"] for p in text_pages)
    known = sum(p["known_words"] for p in text_pages)
    overall = known / checked if checked else 0.0

    lines = [
        "# Качество OCR",
        "",
        "Оценка без эталонной разметки: доля распознанных слов, которые есть в словаре "
        "русского языка (pymorphy3). Скрипт: `evaluate_ocr_quality.py`.",
        "",
        "## Итог",
        "",
        f"- Документов: {len(docs)}, страниц: {len(pages)}.",
        f"- Страниц с текстом (от {MIN_WORDS} слов): {len(text_pages)}.",
        f"- Доля словарных слов по корпусу: **{pct(overall)}** "
        f"({num(known)} из {num(checked)} слов).",
        "",
        "## Страницы по качеству",
        "",
        "| Качество | Критерий | Страниц | Доля |",
        "|---|---|---|---|",
    ]
    crit = {
        "good": f"≥ {int(GOOD * 100)} % словарных слов",
        "fair": f"{int(FAIR * 100)}–{int(GOOD * 100)} %",
        "poor": f"< {int(FAIR * 100)} %",
        "empty": f"< {MIN_WORDS} слов",
    }
    for q in ("good", "fair", "poor", "empty"):
        lines.append(f"| `{q}` | {crit[q]} | {by_q[q]} | {pct(by_q[q] / len(pages))} |")

    lines += ["", "## По типу страницы", "",
              "| Тип страницы | Страниц | Словарных слов | Пустых | Плохих |",
              "|---|---|---|---|---|"]
    by_type = defaultdict(list)
    for p in pages:
        by_type[p["page_type"] or "—"].append(p)
    for t, ps in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        ch = sum(p["checked_words"] for p in ps if p["quality"] != "empty")
        kn = sum(p["known_words"] for p in ps if p["quality"] != "empty")
        lines.append(
            f"| `{t}` | {len(ps)} | {pct(kn / ch) if ch else '—'} | "
            f"{pct(sum(p['quality'] == 'empty' for p in ps) / len(ps))} | "
            f"{pct(sum(p['quality'] == 'poor' for p in ps) / len(ps))} |"
        )

    ranked = sorted([d for d in docs if d["text_pages"]], key=lambda d: d["lexicon_rate"])
    lines += ["", "## Документы: худшие и лучшие", "",
              "| Документ | Страниц | С текстом | Словарных слов | Плохих страниц |",
              "|---|---|---|---|---|"]
    show = ranked[:5] + [None] + ranked[-5:] if len(ranked) > 10 else ranked
    for d in show:
        if d is None:
            lines.append("| … | | | | |")
            continue
        lines.append(f"| {d['document']} | {d['pages']} | {d['text_pages']} | "
                     f"{pct(d['lexicon_rate'])} | {d['poor_pages']} |")

    lines += ["", "## Частые несловарные слова", "",
              "Смесь ошибок OCR, обрывков слов, сокращений, фамилий и топонимов.", ""]
    lines.append(", ".join(f"{w} ({n})" for w, n in worst_words))

    lines += ["", "## Ограничения", "",
              "- Метрика завышает качество: ошибка, которая дала другое настоящее слово "
              "(«хе» вместо «же»), считается верной.",
              "- Метрика занижает качество: фамилии, топонимы и археологические термины, "
              "которых нет в словаре, считаются ошибками.",
              "- Точная оценка требует эталона: ручной расшифровки 20–30 страниц и расчёта "
              "CER/WER (доли ошибочных символов и слов).", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Оценка качества OCR без эталона")
    ap.add_argument("--results", default="results", help="Папка с результатами run_pipeline.py")
    ap.add_argument("--report", default="docs/ocr_quality.md", help="Куда записать сводный отчёт")
    args = ap.parse_args()

    results = Path(args.results)
    morph = pymorphy3.MorphAnalyzer()
    cache = {}
    unknown = Counter()

    pages, docs = [], []
    doc_dirs = sorted(d for d in results.iterdir() if d.is_dir() and (d / "ocr_text").exists())
    for doc_dir in doc_dirs:
        types = load_page_types(doc_dir)
        doc_pages = []
        for txt in sorted((doc_dir / "ocr_text").glob("page_*.txt")):
            page_id = int(re.search(r"(\d+)", txt.stem).group(1))
            text = txt.read_text(encoding="utf-8", errors="ignore")
            m = page_metrics(text, morph, cache)
            if m["quality"] != "empty":
                for t in WORD_RE.findall(HYPHEN_BREAK.sub(r"\1\2", text)):
                    if len(t) >= 3 and CYR_RE.match(t) and not cache.get(t.lower(), True):
                        unknown[t.lower()] += 1
            row = {"document": doc_dir.name, "page_id": page_id,
                   "page_type": types.get(page_id, ""), **m}
            pages.append(row)
            doc_pages.append(row)

        if not doc_pages:
            continue
        tp = [p for p in doc_pages if p["quality"] != "empty"]
        ch = sum(p["checked_words"] for p in tp)
        kn = sum(p["known_words"] for p in tp)
        docs.append({
            "document": doc_dir.name,
            "pages": len(doc_pages),
            "text_pages": len(tp),
            "lexicon_rate": round(kn / ch, 4) if ch else 0.0,
            "good_pages": sum(p["quality"] == "good" for p in doc_pages),
            "fair_pages": sum(p["quality"] == "fair" for p in doc_pages),
            "poor_pages": sum(p["quality"] == "poor" for p in doc_pages),
            "empty_pages": sum(p["quality"] == "empty" for p in doc_pages),
        })

    if not pages:
        raise SystemExit(f"Не найдено OCR-текстов в {results}")

    write_csv(results / "ocr_quality_pages.csv", pages)
    write_csv(results / "ocr_quality_documents.csv", docs)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(pages, docs, unknown.most_common(40)), encoding="utf-8")

    print(f"Документов: {len(docs)}, страниц: {len(pages)}")
    print(f"Отчёт: {report}")
    print(f"Таблицы: {results / 'ocr_quality_pages.csv'}, {results / 'ocr_quality_documents.csv'}")


if __name__ == "__main__":
    main()
