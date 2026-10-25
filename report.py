"""Reporting for saved OCR experiments. Python 3.10+, standard library only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.sax.saxutils import escape

STEPS = (
    ("gray200", "baseline", "Отказ от бинаризации · 200 dpi"),
    ("gray300", "gray200", "Рендер 300 вместо 200 dpi · gray"),
    ("gray300_best", "gray300", "tessdata_best вместо system · 300 dpi"),
)


def number(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("Non-finite number")
    return n


def integer(value):
    n = number(value)
    if n < 0 or not n.is_integer():
        raise ValueError("Expected non-negative integer")
    return int(n)


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Expected True or False")
    return value == "True"


@dataclass(frozen=True)
class MetricDelta:
    a: str
    b: str
    rate_diff_pp: float
    time_diff_s: float


@dataclass(frozen=True)
class VariantMetrics:
    variant: str
    pages: int
    lexicon_rate: float
    sec_per_page: float
    cohort: str

    def __post_init__(self):
        if not self.variant or not self.cohort or type(self.pages) is not int or self.pages <= 0:
            raise ValueError("Variant, cohort and positive page count are required")
        if not math.isfinite(self.lexicon_rate) or not 0 <= self.lexicon_rate <= 1:
            raise ValueError("Rate must be finite and within [0,1]")
        if not math.isfinite(self.sec_per_page) or self.sec_per_page <= 0:
            raise ValueError("Time must be finite and positive")

    def __sub__(self, other):
        if not isinstance(other, VariantMetrics):
            return NotImplemented
        if self.cohort != other.cohort or self.pages != other.pages:
            raise ValueError("Cannot compare different cohorts or page counts")
        return MetricDelta(self.variant, other.variant,
                           100 * (self.lexicon_rate - other.lexicon_rate),
                           self.sec_per_page - other.sec_per_page)


class ExperimentRepository:
    """Loads local aggregate tables; never runs OCR or modifies sources."""
    def __init__(self, directory, cohort):
        self.directory = Path(directory).resolve()
        if not cohort:
            raise ValueError("Explicit cohort identifier required")
        self.cohort = cohort

    def _rows(self, name, required):
        with (self.directory / name).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"{name}: missing columns")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{name}: empty table")
        return rows

    def load(self):
        variants = {}
        for r in self._rows("summary.csv", {"variant", "pages", "lexicon_rate", "sec_per_page"}):
            v = VariantMetrics(r["variant"], integer(r["pages"]), number(r["lexicon_rate"]),
                               number(r["sec_per_page"]), self.cohort)
            if v.variant in variants:
                raise ValueError("Duplicate variant")
            variants[v.variant] = v
        fields = {"a", "b", "pages", "rate_diff_pp", "rate_ci_low", "rate_ci_high",
                  "known_diff_pct", "known_ci_low", "known_ci_high", "pages_better", "pages_worse",
                  "rate_significant", "known_significant"}
        tests = {}
        for r in self._rows("significance.csv", fields):
            key = (r["a"], r["b"])
            if key in tests or key[0] == key[1] or any(k not in variants for k in key):
                raise ValueError("Duplicate or invalid comparison")
            v, b = (variants[k] for k in key)
            delta = v - b
            n = integer(r["pages"])
            if n != v.pages:
                raise ValueError("Paired comparison and summary page counts differ")
            t = {k: number(r[k]) for k in ("rate_diff_pp", "rate_ci_low", "rate_ci_high",
                                           "known_diff_pct", "known_ci_low", "known_ci_high")}
            if abs(delta.rate_diff_pp - t["rate_diff_pp"]) > .021:
                raise ValueError("Saved rate difference inconsistent with rounded summary")
            for prefix in ("rate", "known"):
                lo, hi = t[prefix + "_ci_low"], t[prefix + "_ci_high"]
                if lo > hi:
                    raise ValueError("Reversed confidence interval")
                if boolean(r[prefix + "_significant"]) != (lo > 0 or hi < 0):
                    raise ValueError("Significance flag conflicts with interval")
            better, worse = integer(r["pages_better"]), integer(r["pages_worse"])
            if better + worse > n:
                raise ValueError("Better + worse exceeds paired pages")
            t.update(a=key[0], b=key[1], pages=n, pages_better=better, pages_worse=worse)
            tests[key] = t
        for a, b, _ in STEPS:
            if (a, b) not in tests:
                raise ValueError(f"Missing planned comparison: {a} vs {b}")
        return variants, tests


class SvgFigures:
    """Two vector figures with explicit units; no plotting dependencies."""
    def __init__(self):
        self.parts = []

    def begin(self, title, subtitle):
        self.parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">',
                      '<rect width="1200" height="650" fill="#faf9f5"/>']
        self.text(30, 45, title, 27)
        self.text(30, 78, subtitle, 16, "#666666")

    def text(self, x, y, value, size=17, color="#202728", anchor="start"):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color="#d4d8d4", width=1):
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>')

    def dot(self, x, y, color="#27776e", radius=6):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def write(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")

    def effects(self, tests, path):
        self.begin("Что изменило словарную долю", "Сохранённые парные сравнения; 50 страниц. Точки — разницы, отрезки — 95% бутстреп-ДИ.")
        scale = lambda v: 500 + (v + 3) / 23 * 620
        for tick in (0, 5, 10, 15, 20):
            self.line(scale(tick), 120, scale(tick), 445, "#9ba8a2" if tick == 0 else "#e0e3df")
            self.text(scale(tick), 480, tick, 16, anchor="middle")
        for i, (a, b, label) in enumerate(STEPS):
            y = 175 + i * 115
            t = tests[(a, b)]
            c = "#27776e" if t["rate_ci_low"] > 0 else "#6b7580"
            self.text(30, y - 12, label, 19)
            self.text(30, y + 18, f'{t["rate_diff_pp"]:+.2f} п.п. [{t["rate_ci_low"]:+.2f}; {t["rate_ci_high"]:+.2f}]', 17, c)
            self.line(scale(t["rate_ci_low"]), y, scale(t["rate_ci_high"]), y, c, 4)
            self.dot(scale(t["rate_diff_pp"]), y, c)
        self.text(810, 525, "Изменение словарной доли, п.п.", 17, anchor="middle")
        self.text(30, 585, "Словарная доля ≠ точность OCR. ДИ взяты из significance.csv, не пересчитаны.", 17)
        self.text(30, 618, "5000 перевыборок страниц; без кластеризации по отчёту и поправки на множественные сравнения.", 15, "#666666")
        self.write(path)

    def tradeoff(self, variants, path):
        self.begin("Время обработки и словарная доля", "Все 9 конфигураций на сохранённой выборке. Время — наблюдение этого запуска, не универсальный benchmark.")
        sx = lambda v: 125 + (v - .9) / 2.4 * 970
        sy = lambda v: 510 - (v - 45) / 23 * 350
        for t in (1, 1.5, 2, 2.5, 3):
            self.line(sx(t), 130, sx(t), 510)
            self.text(sx(t), 540, t, 16, anchor="middle")
        for t in (45, 50, 55, 60, 65):
            self.line(125, sy(t), 1095, sy(t))
            self.text(105, sy(t) + 5, t, 16, anchor="end")
        self.text(30, 112, "Словарных слов, %", 18)
        offsets = {"baseline": (12, 20), "gray200": (12, -15), "gray300": (14, -18),
                   "gray300_best": (-15, -15), "gray300_psm4": (-18, -24),
                   "gray300_psm6": (-18, 20), "otsu300": (12, -12),
                   "sauvola300": (12, 24), "gray300_std": (12, -12)}
        for name, v in variants.items():
            x, y = sx(v.sec_per_page), sy(v.lexicon_rate * 100)
            c = "#27776e" if name in ("gray200", "gray300", "gray300_best") else "#8b9297"
            self.dot(x, y, c)
            dx, dy = offsets.get(name, (12, -12))
            self.text(x + dx, y + dy, name, 16, c, "end" if dx < 0 else "start")
        self.text(610, 585, "Среднее время на страницу, с (включает подготовку / рендер)", 18, anchor="middle")
        self.text(30, 625, "Источники: summary.csv; run_experiment.py. Различия времени не проверены статистически.", 15, "#666666")
        self.write(path)


class EvidenceReport:
    """Composition: report owns repository and renderer."""
    def __init__(self, repository, figures):
        self.repository, self.figures = repository, figures

    def build(self, destination):
        dest = Path(destination).resolve()
        src = self.repository.directory
        if dest == src or src in dest.parents or dest in src.parents:
            raise ValueError("Output must not overlap the source directory")
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError("Output directory must be empty; existing results are not overwritten")
        variants, tests = self.repository.load()
        dest.mkdir(parents=True, exist_ok=True)
        changes = [asdict(variants[a] - variants[b]) for a, b, _ in STEPS]
        for change in changes:
            for key in ("rate_diff_pp", "time_diff_s"):
                change[key] = round(change[key], 4)
        with (dest / "changes.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(changes[0]))
            w.writeheader(); w.writerows(changes)
        self.figures.effects(tests, dest / "effects.svg")
        self.figures.tradeoff(variants, dest / "tradeoff.svg")
        info = {"cohort": self.repository.cohort, "variants": len(variants),
                "pages_per_variant": sorted({v.pages for v in variants.values()}),
                "saved_comparisons": len(tests), "intervals": "Imported; not recomputed",
                "inputs": {name: hashlib.sha256((src / name).read_bytes()).hexdigest()
                           for name in ("summary.csv", "significance.csv")}}
        (dest / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--cohort", default="ocr_tuning@b3768a7:saved-50-pages")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = EvidenceReport(ExperimentRepository(args.data, args.cohort), SvgFigures())
    print(json.dumps(report.build(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
