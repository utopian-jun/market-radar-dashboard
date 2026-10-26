import io, os, sys, unittest
from datetime import date
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backfill, build


def ser(vals, start="2025-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


class BackfillTest(unittest.TestCase):
    def test_columns_order_weekend_exclusion_and_source(self):
        s = {"VIX": ser([10 + i for i in range(10)], "2025-10-27")}
        out = backfill.assemble(s, date(2025, 10, 27), date(2025, 11, 7), exclude={"2025-10-29"})
        self.assertEqual(list(out.columns), build.COLUMNS)
        self.assertNotIn("2025-10-29", set(out["수집일"]))
        self.assertTrue(all(pd.Timestamp(d).dayofweek < 5 for d in out["수집일"]))
        self.assertTrue((out["출처"] == "백필").all())
        self.assertTrue(out["잠정판정"].isna().all())          # 판정은 비운다

    def test_200ma_uses_history_before_start(self):
        spy = ser(list(range(1, 301)), "2024-01-01")            # 1..300
        out = backfill.assemble({"SPY": spy}, spy.index[250].date(), spy.index[-1].date(), exclude=set())
        first = out.iloc[0]
        want = spy.iloc[51:251].mean()                          # 시작일 포함 직전 200개
        self.assertAlmostEqual(first["SPY 200일선"], round(want, 2), places=2)

    def test_all_blank_days_dropped(self):
        a = pd.Series([1.0, None, 3.0], index=pd.bdate_range("2025-11-03", periods=3))
        out = backfill.assemble({"VIX": a}, date(2025, 11, 3), date(2025, 11, 5), set())
        self.assertEqual(len(out), 2)


def csv_of(rows):
    return pd.DataFrame(rows, columns=build.COLUMNS).to_csv(index=False)


def row(day, vix, src):
    r = {c: "" for c in build.COLUMNS}
    r.update({"수집일": day, "VIX": vix, "출처": src})
    return [r[c] for c in build.COLUMNS]


class BuildMergeTest(unittest.TestCase):
    def test_collected_row_wins_over_backfill_same_date(self):
        text = csv_of([row("2026-10-20", "15", "백필"), row("2026-10-21", "99", "백필"), row("2026-10-21", "20", "수집")])
        df = build.parse(text)
        self.assertEqual(len(df), 2)
        self.assertEqual(float(df[df["수집일"] == "2026-10-21"]["VIX"].iloc[0]), 20.0)

    def test_counts_and_stats_include_backfill(self):
        text = csv_of([row("2026-09-01", "10", "백필"), row("2026-09-02", "12", "백필"), row("2026-10-21", "20", "수집")])
        d = build.build_data(build.parse(text), today=date(2026, 10, 22))
        self.assertEqual((d["collection_days"], d["backfill_days"], d["history_days"]), (1, 2, 3))
        self.assertEqual(d["indicators"]["VIX"]["n"], 3)
        self.assertEqual(d["indicators"]["VIX"]["prev_30d"], 12.0)     # 30일 전 값이 첫날부터 존재
        self.assertEqual(d["series"]["source"], ["백필", "백필", "수집"])
        self.assertEqual(d["collection_first_date"], "2026-10-21")

    def test_streak_stops_at_blank_backfill_rows(self):
        r1, r2 = row("2026-10-20", "15", "백필"), row("2026-10-21", "20", "수집")
        r2[build.COLUMNS.index("잠정판정")] = "중립"
        r3 = row("2026-10-22", "21", "수집"); r3[build.COLUMNS.index("잠정판정")] = "중립"
        d = build.build_data(build.parse(csv_of([r1, r2, r3])), today=date(2026, 10, 23))
        self.assertEqual(d["verdict"]["provisional_streak_days"], 2)

    def test_legacy_22_column_header_is_all_collected(self):
        text = csv_of([row("2026-10-21", "20", "")]).split("\n")
        text = "\n".join(l.rsplit(",", 1)[0] for l in text)
        d = build.build_data(build.parse(text), today=date(2026, 10, 22))
        self.assertEqual((d["collection_days"], d["backfill_days"]), (1, 0))

    def test_wrong_header_still_fails(self):
        bad = csv_of([row("2026-10-21", "20", "")]).replace("VIX", "VXX", 1)
        with self.assertRaises(build.SchemaError):
            build.parse(bad)


if __name__ == "__main__":
    unittest.main()
