from pathlib import Path
import csv
import json
import re
from collections import Counter


INPUT_MANIFEST = Path("ocr_reports_manifest.csv")
REPORT_STATS_CSV = Path("report_stats.csv")
TERM_FREQ_CSV = Path("term_frequencies.csv")
CORPUS_STATS_JSON = Path("corpus_stats.json")

MIN_TOKEN_LEN = 3

STOPWORDS = {
    "это", "как", "для", "или", "при", "под", "над", "без", "что", "его", "ее",
    "её", "они", "она", "оно", "мы", "вы", "ты", "так", "уже", "еще", "ещё",
    "был", "была", "были", "быть", "есть", "нет", "где", "все", "всё", "из",
    "на", "по", "от", "до", "за", "не", "ни", "но", "а", "и", "в", "во", "с",
    "со", "к", "ко", "о", "об", "обо", "у", "то", "же", "ли", "их", "мы", "он",
    "она", "они", "the", "and", "for", "with", "this", "that", "from", "are",
    "was", "were", "into", "und", "der", "die", "das"
}


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text)

    clean_tokens = []
    for token in tokens:
        if len(token) < MIN_TOKEN_LEN:
            continue
        if token in STOPWORDS:
            continue
        clean_tokens.append(token)

    return clean_tokens


def main():
    if not INPUT_MANIFEST.exists():
        print(f"ERROR: input manifest not found: {INPUT_MANIFEST}")
        return

    report_rows = []
    corpus_counter = Counter()

    total_reports = 0
    total_chars = 0
    total_words = 0
    total_pages = 0

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_reports += 1

            report_id = row["report_id"]
            text_path = Path(row["report_text_path"])
            page_count = int(row["page_count"])
            char_count = int(row["char_count"])
            word_count = int(row["word_count"])
            quality_label = row["quality_label"]

            text = read_text(text_path)
            tokens = tokenize(text)
            token_count_clean = len(tokens)

            unique_terms = len(set(tokens))
            corpus_counter.update(tokens)

            total_chars += char_count
            total_words += word_count
            total_pages += page_count

            report_rows.append({
                "report_id": report_id,
                "report_text_path": str(text_path.resolve()),
                "page_count": page_count,
                "char_count": char_count,
                "word_count": word_count,
                "quality_label": quality_label,
                "clean_token_count": token_count_clean,
                "unique_terms": unique_terms,
            })

            print(f"[{total_reports}] Analyzed report: {report_id}")

    if not report_rows:
        print("No report rows analyzed.")
        return

    with open(REPORT_STATS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "report_text_path",
                "page_count",
                "char_count",
                "word_count",
                "quality_label",
                "clean_token_count",
                "unique_terms",
            ]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    with open(TERM_FREQ_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["term", "count"]
        )
        writer.writeheader()
        for term, count in corpus_counter.most_common():
            writer.writerow({
                "term": term,
                "count": count,
            })

    corpus_stats = {
        "report_count": total_reports,
        "page_count_total": total_pages,
        "char_count_total": total_chars,
        "word_count_total": total_words,
        "avg_pages_per_report": round(total_pages / total_reports, 3),
        "avg_words_per_report": round(total_words / total_reports, 3),
        "vocab_size": len(corpus_counter),
        "top_20_terms": corpus_counter.most_common(20),
    }

    with open(CORPUS_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, ensure_ascii=False, indent=2)

    print(f"Done. Reports analyzed: {total_reports}")
    print(f"Report stats written to: {REPORT_STATS_CSV}")
    print(f"Term frequencies written to: {TERM_FREQ_CSV}")
    print(f"Corpus stats written to: {CORPUS_STATS_JSON}")


if __name__ == "__main__":
    main()