"""
Эталонная выборка страниц для расчёта CER/WER.

Шаги:
  1. Отчёты: стратифицированная случайная выборка по каталогу архива
     (reports_catalog.csv), слои — год работ; 30 основных и по 2 запасных
     на слой, seed 2026. Размер слоя пропорционален числу отчётов в нём.
  2. Кандидаты: по 8 случайных страниц из каждого отчёта (seed 7), без страниц,
     уже использованных в experiments/ocr_tuning.
  3. Выбор страницы (вручную, по миниатюрам; см. PICKS): из кандидатов берётся
     страница с текстом нужного типа. Если среди кандидатов текста нет, просмотрен
     весь отчёт. Если в отчёте текста нет вообще (альбом из фотографий
     и рисунков), отчёт заменяется первым запасным из того же слоя. В альбомах
     с подписями к иллюстрациям берётся страница с подписями.
  4. Извлечение исходного скана страницы без пересжатия (pdfimages -all).

Запуск из корня проекта:
    python ground_truth/make_sample.py            # шаги 1–2, печать кандидатов
    python ground_truth/make_sample.py --extract  # шаг 4 по PICKS
"""

import argparse
import csv
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "input_reports"
POPPLER = ROOT / "tools" / "poppler-26.02.0" / "Library" / "bin"

N_REPORTS, RESERVE, SEED_REPORTS = 30, 2, 2026
N_CAND, SEED_PAGES = 8, 7

# Отчёт (номер по каталогу), страница PDF, слой, тип страницы, основание выбора.
PICKS = [
    ("Р-1_№ 29", 10, "1944–1945", "машинопись", "основной"),
    ("Р-1_№ 26", 3, "1944–1945", "машинопись", "основной"),
    ("Р-1_№ 28", 9, "1944–1945", "машинопись", "замена Р-1 № 198: нет текстовых страниц"),
    ("Р-1_№ 44", 19, "1946", "машинопись", "основной"),
    ("Р-1_№ 65", 16, "1946", "рукопись", "основной"),
    ("Р-1_№ 85", 15, "1946", "машинопись", "основной"),
    ("Р-1_№ 62", 10, "1946", "машинопись", "основной"),
    ("Р-1_№ 46", 11, "1946", "машинопись", "замена Р-1 № 54: только рисунки"),
    ("Р-1_№ 129", 8, "1947", "машинопись", "замена Р-1 № 130а: альбом без подписей"),
    ("Р-1_№ 176", 12, "1947", "подпись", "замена Р-1 № 164: альбом без текста"),
    ("Р-1_№ 122а", 2, "1947", "машинопись", "основной"),
    ("Р-1_№ 128", 8, "1947", "машинопись", "основной"),
    ("Р-1_№ 166", 7, "1947", "машинопись", "основной"),
    ("Р-1_№ 158", 5, "1947", "подпись", "основной, альбом"),
    ("Р-1_№ 121", 7, "1947", "машинопись", "основной"),
    ("Р-1_№ 243", 41, "1948", "машинопись", "основной"),
    ("Р-1_№ 224", 30, "1948", "таблица", "основной"),
    ("Р-1_№ 265", 6, "1948", "машинопись", "основной"),
    ("Р-1_№ 213", 10, "1948", "машинопись", "основной"),
    ("Р-1_№ 279", 25, "1948", "машинопись", "основной"),
    ("Р-1_№ 257", 9, "1948", "машинопись", "основной"),
    ("Р-1_№ 349", 10, "1949", "машинопись", "основной"),
    ("Р-1_№ 319", 29, "1949", "машинопись", "основной"),
    ("Р-1_№ 369а", 5, "1949", "список", "основной"),
    ("Р-1_№ 309", 3, "1949", "машинопись", "основной"),
    ("Р-1_№ 326", 29, "1949", "рукопись", "основной"),
    ("Р-1_№ 357", 9, "1949", "подпись", "основной, альбом"),
    ("Р-1_№ 327", 7, "1949", "подпись", "основной, альбом"),
    ("Р-1_№ 300", 12, "нет года", "рукопись", "основной"),
    ("Р-1_№ 305", 19, "нет года", "машинопись", "основной"),
]


def tool(name):
    exe = POPPLER / (name + ".exe")
    return str(exe) if exe.exists() else name


def pdf_for(filename):
    found = sorted(INPUT.glob(f"*_{filename}.pdf"))
    if not found:
        raise SystemExit(f"Нет PDF для {filename} в {INPUT}")
    return found[0]


def stratum(year):
    if not year.isdigit():
        return "нет года"
    return "1944–1945" if int(year) <= 1945 else year


def sample_reports():
    with open(ROOT / "reports_catalog.csv", encoding="utf-8-sig") as f:
        cat = list(csv.DictReader(f))
    for c in cat:
        c["stratum"] = stratum(c["year"])
    cnt = Counter(c["stratum"] for c in cat)
    quota = {s: N_REPORTS * k / len(cat) for s, k in cnt.items()}
    alloc = {s: int(v) for s, v in quota.items()}
    rest = N_REPORTS - sum(alloc.values())
    for s, _ in sorted(quota.items(), key=lambda x: -(x[1] - int(x[1])))[:rest]:
        alloc[s] += 1
    rng = random.Random(SEED_REPORTS)
    chosen = []
    for s in sorted(alloc):
        pool = [c for c in cat if c["stratum"] == s]
        rng.shuffle(pool)
        for i, c in enumerate(pool[: alloc[s] + RESERVE]):
            chosen.append((s, i >= alloc[s], c["filename"]))
    return chosen


def page_count(pdf):
    out = subprocess.run([tool("pdfinfo"), str(pdf)], capture_output=True, text=True).stdout
    return int(next(l.split()[-1] for l in out.splitlines() if l.startswith("Pages:")))


def used_pages():
    path = ROOT / "experiments" / "ocr_tuning" / "pages.csv"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {(r["document"], int(r["page_id"])) for r in csv.DictReader(f)}


def candidates(chosen):
    used, rng, rows = used_pages(), random.Random(SEED_PAGES), []
    for s, reserve, filename in chosen:
        pdf = pdf_for(filename)
        doc = pdf.stem.replace(" ", "_")
        pool = [p for p in range(1, page_count(pdf) + 1) if (doc, p) not in used]
        for p in sorted(rng.sample(pool, min(N_CAND, len(pool)))):
            rows.append((s, reserve, filename, p))
    return rows


def extract():
    img_dir = HERE / "images"
    img_dir.mkdir(exist_ok=True)
    out = []
    for filename, page, s, kind, note in PICKS:
        pdf = pdf_for(filename)
        stem = f"{pdf.stem[:4]}_p{page:03d}"
        info = subprocess.run([tool("pdfimages"), "-list", "-f", str(page), "-l", str(page), str(pdf)],
                              capture_output=True, text=True).stdout.splitlines()[2:]
        big = max((l.split() for l in info), key=lambda r: int(r[3]) * int(r[4]))
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([tool("pdfimages"), "-all", "-f", str(page), "-l", str(page),
                            str(pdf), str(Path(tmp) / "x")], check=True)
            src = max(Path(tmp).iterdir(), key=lambda q: q.stat().st_size)
            if src.suffix in (".jpg", ".png"):
                dst = img_dir / (stem + src.suffix)
                shutil.copy(src, dst)
            else:
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = None
                dst = img_dir / (stem + ".png")
                Image.open(src).save(dst)
        out.append({"id": stem, "report": filename, "file": pdf.name, "page": page,
                    "stratum": s, "page_type": kind, "selection": note,
                    "image": f"images/{dst.name}",
                    "width": big[3], "height": big[4], "ppi": big[12]})
        print(stem, dst.name)
    with open(HERE / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()
    if args.extract:
        extract()
        return
    for s, reserve, filename, p in candidates(sample_reports()):
        print(s, "запасной" if reserve else "основной", filename, p, sep="\t")


if __name__ == "__main__":
    main()
