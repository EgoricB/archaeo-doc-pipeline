import csv
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree
from research_reporting.report import VariantMetrics, ExperimentRepository, SvgFigures, EvidenceReport

DATA = Path(__file__).parent / "data"

class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        shutil.copytree(DATA, self.data)
        self.repo = ExperimentRepository(self.data, "saved-50")

    def mutate(self, name, change):
        path = self.data / name
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        change(rows)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def test_load(self):
        variants, tests = self.repo.load()
        self.assertEqual((len(variants), len(tests)), (9, 21))
        self.assertEqual(variants["baseline"].pages, 50)

    def test_subtraction_used_for_effect(self):
        v, _ = self.repo.load()
        d = v["gray200"] - v["baseline"]
        self.assertAlmostEqual(d.rate_diff_pp, 15.70)
        self.assertAlmostEqual(d.time_diff_s, -.63)

    def test_unsupported_operand(self):
        v, _ = self.repo.load()
        self.assertIs(v["baseline"].__sub__(1), NotImplemented)
        with self.assertRaises(TypeError): v["baseline"] - 1

    def test_cohort_mismatch(self):
        with self.assertRaises(ValueError):
            VariantMetrics("a", 50, .5, 1, "x") - VariantMetrics("b", 50, .4, 1, "y")

    def test_page_mismatch(self):
        with self.assertRaises(ValueError):
            VariantMetrics("a", 50, .5, 1, "x") - VariantMetrics("b", 49, .4, 1, "x")

    def test_invalid_metrics(self):
        for rate, secs, pages in [(float("nan"),1,50),(1.1,1,50),(.5,0,50),(.5,float("inf"),50),(.5,1,0)]:
            with self.subTest(rate=rate, secs=secs, pages=pages), self.assertRaises(ValueError):
                VariantMetrics("a", pages, rate, secs, "x")

    def test_empty_table(self):
        self.mutate("summary.csv", lambda rows: rows.clear())
        with self.assertRaises(ValueError): self.repo.load()

    def test_duplicate_variant(self):
        self.mutate("summary.csv", lambda rows: rows.append(rows[0].copy()))
        with self.assertRaises(ValueError): self.repo.load()

    def test_missing_columns(self):
        (self.data / "summary.csv").write_text("variant\na\n")
        with self.assertRaises(ValueError): self.repo.load()

    def test_fractional_pages(self):
        self.mutate("summary.csv", lambda r: r[0].update(pages="49.5"))
        with self.assertRaises(ValueError): self.repo.load()

    def test_inconsistent_point(self):
        self.mutate("significance.csv", lambda r: r[0].update(rate_diff_pp="99"))
        with self.assertRaises(ValueError): self.repo.load()

    def test_reversed_interval(self):
        self.mutate("significance.csv", lambda r: r[0].update(rate_ci_low="30"))
        with self.assertRaises(ValueError): self.repo.load()

    def test_conflicting_flag(self):
        self.mutate("significance.csv", lambda r: r[0].update(rate_significant="False"))
        with self.assertRaises(ValueError): self.repo.load()

    def test_invalid_pair_counts(self):
        self.mutate("significance.csv", lambda r: r[0].update(pages_better="50", pages_worse="1"))
        with self.assertRaises(ValueError): self.repo.load()

    def test_missing_planned_comparison(self):
        self.mutate("significance.csv", lambda r: r.pop(0))
        with self.assertRaises(ValueError): self.repo.load()

    def test_duplicate_comparison(self):
        self.mutate("significance.csv", lambda r: r.append(r[0].copy()))
        with self.assertRaises(ValueError): self.repo.load()

    def test_build_and_source_integrity(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.iterdir()}
        dest = self.root / "derived"
        info = EvidenceReport(self.repo, SvgFigures()).build(dest)
        self.assertEqual(info["variants"], 9)
        for name in ("effects.svg", "tradeoff.svg"):
            self.assertEqual(ElementTree.parse(dest / name).getroot().tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.data.iterdir()})
        self.assertTrue((dest / "changes.csv").exists())
        with self.assertRaises(FileExistsError): EvidenceReport(self.repo, SvgFigures()).build(dest)

    def test_source_overlap_refused(self):
        for dest in (self.data, self.data / "derived", self.root):
            with self.subTest(dest=dest), self.assertRaises(ValueError):
                EvidenceReport(self.repo, SvgFigures()).build(dest)

if __name__ == "__main__":
    unittest.main()
