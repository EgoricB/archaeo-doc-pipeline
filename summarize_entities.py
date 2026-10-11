from pathlib import Path
import csv
from collections import Counter, defaultdict


INPUT_CSV = Path("entities.csv")
ENTITY_FREQ_CSV = Path("entity_frequencies.csv")
ENTITY_BY_REPORT_CSV = Path("entity_by_report.csv")
ENTITY_TYPE_SUMMARY_CSV = Path("entity_type_summary.csv")


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: input file not found: {INPUT_CSV}")
        return

    all_counter = Counter()
    type_counter = Counter()
    by_report_counter = Counter()

    rows_total = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_total += 1

            report_id = row["report_id"]
            entity_type = row["entity_type"]
            entity_value = row["entity_value"]

            all_counter[(entity_type, entity_value)] += 1
            type_counter[entity_type] += 1
            by_report_counter[(report_id, entity_type)] += 1

    if rows_total == 0:
        print("No entity rows found.")
        return

    with open(ENTITY_FREQ_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_type", "entity_value", "count"]
        )
        writer.writeheader()

        for (entity_type, entity_value), count in all_counter.most_common():
            writer.writerow({
                "entity_type": entity_type,
                "entity_value": entity_value,
                "count": count,
            })

    with open(ENTITY_BY_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["report_id", "entity_type", "count"]
        )
        writer.writeheader()

        for (report_id, entity_type), count in sorted(by_report_counter.items()):
            writer.writerow({
                "report_id": report_id,
                "entity_type": entity_type,
                "count": count,
            })

    with open(ENTITY_TYPE_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_type", "count"]
        )
        writer.writeheader()

        for entity_type, count in type_counter.most_common():
            writer.writerow({
                "entity_type": entity_type,
                "count": count,
            })

    print(f"Done. Rows processed: {rows_total}")
    print(f"Entity frequencies written to: {ENTITY_FREQ_CSV}")
    print(f"Entity by report written to: {ENTITY_BY_REPORT_CSV}")
    print(f"Entity type summary written to: {ENTITY_TYPE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()