from pathlib import Path
import csv
import re


INPUT_MANIFEST = Path("ocr_reports_manifest.csv")
OUTPUT_CSV = Path("entities.csv")


ARCH_TERMS = {
    "археология",
    "археологический",
    "археологические",
    "археологических",
    "памятник",
    "памятники",
    "поселение",
    "поселения",
    "стоянка",
    "стоянки",
    "городище",
    "городища",
    "могильник",
    "могильники",
    "курган",
    "курганы",
    "некрополь",
    "раскоп",
    "раскопки",
    "шурф",
    "шурфы",
    "керамика",
    "фрагмент",
    "фрагменты",
    "слой",
    "слои",
    "квадрат",
    "квадраты",
    "ямы",
    "яма",
    "погребение",
    "погребения",
    "инвентарь",
    "культурный",
    "культурного",
    "стратиграфия",
    "стратиграфический",
    "находка",
    "находки",
    "объект",
    "объекты",
}


TOPONYM_STOPWORDS = {
    "Отчет", "Отчёт", "Глава", "Раздел", "Таблица", "Рисунок", "Приложение",
    "Страница", "Институт", "Экспедиция", "Фото", "Архив", "Район", "Область",
    "Республика", "России", "СССР", "РФ", "Федерации"
}


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_years(text: str):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return sorted(set(years))


def extract_arch_terms(text: str):
    tokens = re.findall(r"[А-Яа-яЁёA-Za-z]+", text.lower())
    found = [tok for tok in tokens if tok in ARCH_TERMS]
    return sorted(set(found))


def extract_toponym_candidates(text: str):
    candidates = re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", text)

    filtered = []
    for cand in candidates:
        if cand in TOPONYM_STOPWORDS:
            continue
        filtered.append(cand)

    counts = {}
    for cand in filtered:
        counts[cand] = counts.get(cand, 0) + 1

    # оставляем только достаточно частые кандидаты
    result = [k for k, v in counts.items() if v >= 2]
    return sorted(result)


def add_entity(rows, report_id, entity_type, entity_value):
    rows.append({
        "report_id": report_id,
        "entity_type": entity_type,
        "entity_value": entity_value,
    })


def main():
    if not INPUT_MANIFEST.exists():
        print(f"ERROR: input manifest not found: {INPUT_MANIFEST}")
        return

    entity_rows = []
    total_reports = 0

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_reports += 1

            report_id = row["report_id"]
            text_path = Path(row["report_text_path"])
            text = read_text(text_path)

            years = extract_years(text)
            arch_terms = extract_arch_terms(text)
            toponyms = extract_toponym_candidates(text)

            for year in years:
                add_entity(entity_rows, report_id, "year", year)

            for term in arch_terms:
                add_entity(entity_rows, report_id, "arch_term", term)

            for topo in toponyms:
                add_entity(entity_rows, report_id, "toponym_candidate", topo)

            print(
                f"[{total_reports}] {report_id}: "
                f"years={len(years)}, arch_terms={len(arch_terms)}, toponyms={len(toponyms)}"
            )

    if not entity_rows:
        print("No entities extracted.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "entity_type", "entity_value"]
        )
        writer.writeheader()
        writer.writerows(entity_rows)

    print(f"Done. Reports processed: {total_reports}")
    print(f"Entities written to: {OUTPUT_CSV}")
    print(f"Total extracted rows: {len(entity_rows)}")


if __name__ == "__main__":
    main()