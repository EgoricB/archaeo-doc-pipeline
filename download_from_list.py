"""Скачивание PDF-отчётов из открытого архива ИА РАН.

Источник списка — официальный каталог архива:
    https://archaeolog.ru/media/reports/index.xml
(из него строится таблица на странице https://archaeolog.ru/pub/elektronnye-polevye-otchety).

    python download_from_list.py                  # скачать всё, чего нет в input_reports/
    python download_from_list.py --update-list    # сначала дописать в report_urls.txt новые отчёты из каталога

Новые ссылки дописываются в конец report_urls.txt, поэтому номера (префиксы имён)
уже скачанных файлов не меняются. Файл сначала качается во временный *.part и
переименовывается только после полной загрузки, так что оборванная загрузка
не принимается за готовый файл. Неудачные загрузки повторяются.
"""

import argparse
import csv
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, unquote, quote

import requests

URLS_FILE = Path("report_urls.txt")
OUTPUT_DIR = Path("input_reports")
CATALOG_URL = "https://archaeolog.ru/media/reports/index.xml"
CATALOG_CSV = Path("reports_catalog.csv")
REPORT_URL = "http://www.archaeolog.ru/media/reports/{}.pdf"
RETRIES = 5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def sanitize_filename(name: str) -> str:
    name = unquote(name)  # декодируем %D0...
    name = name.replace("/", "_").replace("\\", "_")
    # заменяем запрещённые в Windows символы
    name = re.sub(r'[<>:"|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def filename_from_url(url: str, index: int) -> str:
    parsed = urlparse(url)
    raw_name = Path(parsed.path).name or f"report_{index:04d}.pdf"
    name = sanitize_filename(raw_name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return f"{index:04d}_{name}"


def read_urls():
    if not URLS_FILE.exists():
        return []
    return [l.strip() for l in URLS_FILE.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def fetch_catalog():
    """Каталог архива: список словарей с шифром, автором, названием, годом, числом страниц."""
    r = requests.get(CATALOG_URL, headers=HEADERS, timeout=120)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    fields = ["serial", "code", "filename", "author", "title", "year", "expedition", "pagescount", "filesizebytes"]
    return [{k: (rep.findtext(k) or "").strip() for k in fields} for rep in root.findall("report")]


def update_list():
    """Дописывает в report_urls.txt отчёты из каталога, которых в нём нет, и сохраняет каталог в CSV."""
    catalog = fetch_catalog()
    with open(CATALOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(catalog[0]))
        w.writeheader()
        w.writerows(catalog)
    print(f"Каталог архива: {len(catalog)} отчётов -> {CATALOG_CSV}")

    urls = read_urls()
    known = {unquote(Path(urlparse(u).path).stem) for u in urls}
    new = [REPORT_URL.format(quote(c["filename"])) for c in catalog if c["filename"] not in known]
    if new:
        text = URLS_FILE.read_text(encoding="utf-8-sig") if URLS_FILE.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        URLS_FILE.write_text(text + "".join(u + "\n" for u in new), encoding="utf-8")
    print(f"Новых отчётов в каталоге: {len(new)}")
    for u in new:
        print("  +", unquote(u))
    missing_in_catalog = known - {c["filename"] for c in catalog}
    if missing_in_catalog:
        print(f"В списке, но нет в каталоге ({len(missing_in_catalog)}): {sorted(missing_in_catalog)}")


def download_file(url: str, dest: Path):
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, headers=HEADERS, timeout=120, stream=True) as r:
        r.raise_for_status()
        if "pdf" not in r.headers.get("Content-Type", "").lower():
            raise ValueError(f"сервер вернул не PDF: {r.headers.get('Content-Type')}")
        expected = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    size = tmp.stat().st_size
    if expected and size != expected:
        raise IOError(f"загрузка оборвалась: {size} из {expected} байт")
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-list", action="store_true", help="Дописать в список новые отчёты из каталога архива")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.update_list:
        update_list()

    urls = read_urls()
    if not urls:
        print(f"Нет ссылок в {URLS_FILE}. Запустите с --update-list")
        return
    print(f"Ссылок: {len(urls)}")

    failed = []
    for i, url in enumerate(urls, start=1):
        dest = OUTPUT_DIR / filename_from_url(url, i)
        if dest.exists():
            continue
        for attempt in range(1, RETRIES + 1):
            try:
                print(f"[DL]  {i}/{len(urls)} {dest.name} (попытка {attempt})")
                download_file(url, dest)
                print(f"[OK]  {dest.name} ({dest.stat().st_size / 1e6:.1f} МБ)")
                break
            except Exception as e:
                print(f"[ERR] {dest.name}: {e}")
                time.sleep(5 * attempt)
        else:
            failed.append(dest.name)
        time.sleep(0.5)

    have = sum(1 for i, u in enumerate(urls, 1) if (OUTPUT_DIR / filename_from_url(u, i)).exists())
    print(f"\nСкачано {have} из {len(urls)}.")
    if failed:
        print(f"Не удалось скачать ({len(failed)}), запустите скрипт ещё раз:")
        for n in failed:
            print("  -", n)


if __name__ == "__main__":
    main()
