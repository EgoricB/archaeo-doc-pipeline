from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FILE_CANDIDATES = {
    "page_analysis": ["page_analysis.csv", "page_analysis-3.csv"],
    "report_stats": ["report_stats.csv", "report_stats-8.csv"],
    "entity_type_summary": ["entity_type_summary.csv", "entity_type_summary-11.csv"],
    "entity_frequencies": ["entity_frequencies.csv", "entity_frequencies-10.csv"],
}


def resolve_existing_file(candidates):
    for name in candidates:
        path = Path(name)
        if path.exists():
            return path
    raise FileNotFoundError("Не найден ни один из файлов: " + ", ".join(candidates))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )
    return df


def load_named_csv(key: str) -> pd.DataFrame:
    path = resolve_existing_file(FILE_CANDIDATES[key])
    df = pd.read_csv(path)
    df = normalize_columns(df)
    print(f"[OK] {key}: {path}")
    print(f"     columns: {list(df.columns)}")
    print(f"     rows: {len(df)}")
    return df


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.family"] = "DejaVu Sans"


def show_barplot(series, title, xlabel, ylabel,
                 color="#2a6f97", rotate_x=False, horizontal=False):
    if series is None or len(series) == 0:
        print(f"[SKIP] Нет данных для графика: {title}")
        return

    plt.figure(figsize=(10, 6))

    if horizontal:
        series = series.sort_values(ascending=True)
        bars = plt.barh(series.index.astype(str), series.values, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        max_val = max(series.values) if len(series.values) else 0
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + (max_val * 0.01 if max_val else 0.1),
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va="center",
                fontsize=9
            )
    else:
        bars = plt.bar(series.index.astype(str), series.values, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        max_val = max(series.values) if len(series.values) else 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max_val * 0.01 if max_val else 0.1),
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.title(title)

    if rotate_x:
        plt.xticks(rotation=40, ha="right")

    plt.tight_layout()
    plt.show()


def plot_page_type_distribution(page_analysis: pd.DataFrame):
    if "pagetype" not in page_analysis.columns:
        print("[SKIP] В page_analysis нет колонки pagetype")
        return

    series = page_analysis["pagetype"].astype(str).value_counts()
    print("\n[INFO] Распределение типов страниц:")
    print(series)

    show_barplot(
        series=series,
        title="Распределение типов страниц",
        xlabel="Тип страницы",
        ylabel="Количество страниц",
        color="#277da1",
        rotate_x=True
    )


def plot_report_sizes(report_stats: pd.DataFrame):
    required = {"reportid", "pagecount"}
    if not required.issubset(report_stats.columns):
        print("[SKIP] В report_stats нет нужных колонок")
        return

    df = report_stats.copy()
    df["pagecount"] = pd.to_numeric(df["pagecount"], errors="coerce")
    df = df.dropna(subset=["pagecount"])
    df = df.sort_values("pagecount", ascending=False)

    series = pd.Series(
        df["pagecount"].astype(int).values,
        index=df["reportid"].astype(str)
    )

    print("\n[INFO] Размеры отчётов:")
    print(series)

    show_barplot(
        series=series,
        title="Количество страниц по отчётам",
        xlabel="ID отчёта",
        ylabel="Число страниц",
        color="#e76f51",
        rotate_x=True
    )


def plot_entity_type_distribution(entity_type_summary: pd.DataFrame):
    required = {"entitytype", "count"}
    if not required.issubset(entity_type_summary.columns):
        print("[SKIP] В entity_type_summary нет нужных колонок")
        return

    df = entity_type_summary.copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])

    series = pd.Series(
        df["count"].astype(int).values,
        index=df["entitytype"].astype(str)
    )

    print("\n[INFO] Типы сущностей:")
    print(series)

    show_barplot(
        series=series,
        title="Распределение типов сущностей",
        xlabel="Тип сущности",
        ylabel="Количество",
        color="#43aa8b",
        rotate_x=True
    )


def plot_top_years(entity_frequencies: pd.DataFrame):
    required = {"entitytype", "entityvalue", "count"}
    if not required.issubset(entity_frequencies.columns):
        print("[SKIP] В entity_frequencies нет нужных колонок")
        return

    df = entity_frequencies.copy()
    df = df[df["entitytype"].astype(str) == "year"].copy()

    if df.empty:
        print("[SKIP] Нет строк с entitytype=year")
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])
    df["entityvalue"] = df["entityvalue"].astype(str)
    df = df.sort_values("count", ascending=False).head(15)

    series = pd.Series(
        df["count"].astype(int).values,
        index=df["entityvalue"]
    )

    print("\n[INFO] Топ годов:")
    print(series)

    show_barplot(
        series=series,
        title="Наиболее частотные годы",
        xlabel="Год",
        ylabel="Количество",
        color="#6a4c93",
        rotate_x=True
    )


def plot_top_arch_terms(entity_frequencies: pd.DataFrame):
    required = {"entitytype", "entityvalue", "count"}
    if not required.issubset(entity_frequencies.columns):
        print("[SKIP] В entity_frequencies нет нужных колонок")
        return

    df = entity_frequencies.copy()
    df = df[df["entitytype"].astype(str) == "archterm"].copy()

    if df.empty:
        print("[SKIP] Нет строк с entitytype=archterm")
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])
    df["entityvalue"] = df["entityvalue"].astype(str)
    df = df.sort_values("count", ascending=False).head(15)

    series = pd.Series(
        df["count"].astype(int).values,
        index=df["entityvalue"]
    )

    print("\n[INFO] Топ археологических терминов:")
    print(series)

    show_barplot(
        series=series,
        title="Топ археологических терминов",
        xlabel="Количество",
        ylabel="Термин",
        color="#8c564b",
        horizontal=True
    )


def main():
    setup_style()

    page_analysis = load_named_csv("page_analysis")
    report_stats = load_named_csv("report_stats")
    entity_type_summary = load_named_csv("entity_type_summary")
    entity_frequencies = load_named_csv("entity_frequencies")

    plot_page_type_distribution(page_analysis)
    plot_report_sizes(report_stats)
    plot_entity_type_distribution(entity_type_summary)
    plot_top_years(entity_frequencies)
    plot_top_arch_terms(entity_frequencies)

    print("\nГотово: графики показаны через matplotlib.")


if __name__ == "__main__":
    main()