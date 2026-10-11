import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

URLS_FILE = Path("report_urls.txt")
OUTPUT_DIR = Path("input_reports")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def download_file(url: str, dest: Path):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    with requests.get(url, headers=headers, timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main():
    ensure_dir(OUTPUT_DIR)

    if not URLS_FILE.exists():
        print(f"URLs file not found: {URLS_FILE}")
        return

    urls = []
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        urls.append(line)

    print(f"Found {len(urls)} URLs")

    for i, url in enumerate(urls, start=1):
        filename = filename_from_url(url, i)
        dest = OUTPUT_DIR / filename

        if dest.exists():
            print(f"[SKIP] {i}/{len(urls)} {filename} (already exists)")
            continue

        try:
            print(f"[DL]   {i}/{len(urls)} {url} -> {filename}")
            download_file(url, dest)
            size_kb = round(dest.stat().st_size / 1024, 2)
            print(f"[OK]   {filename} ({size_kb} KB)")
        except Exception as e:
            print(f"[ERR]  {i}/{len(urls)} {url} -> {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()