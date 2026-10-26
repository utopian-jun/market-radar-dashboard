import json, os, sys, tempfile, unittest
from unittest import mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llm

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "sample", "radar_log.data.json"), encoding="utf-8"))
GOOD = {
    "overall": ["HY OAS는 4.34로 주의 구간이고 평균 3.99보다 높다.", "심리는 극공포인데 신용은 평온하다."],
    "indicators": {k: "현재 구간이고 30일 방향은 위다." for k in llm.KEYS},
    "watch": ["VIX가 20을 넘어서면 변동성 층이 바뀔 수 있다."],
}


def caller_returning(*outs):
    it = iter(outs)
    m = mock.Mock(side_effect=lambda *a, **k: next(it))
    return m


class LlmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["GEMINI_API_KEY"] = "test-key"
        llm.RETRY_WAITS = (0, 0, 0)         # 테스트에서는 기다리지 않는다

    def run_llm(self, caller, **kw):
        return llm.run(DATA, self.tmp, caller=caller, **kw)

    def test_same_day_calls_api_once(self):
        c = caller_returning(json.dumps(GOOD))
        self.assertEqual(self.run_llm(c), "generated")
        self.assertEqual(self.run_llm(c), "cached")
        self.assertEqual(c.call_count, 1)

    def test_no_key_skips_and_writes_nothing(self):
        del os.environ["GEMINI_API_KEY"]
        c = caller_returning()
        self.assertEqual(self.run_llm(c), "skipped-no-key")
        self.assertEqual(c.call_count, 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "commentary.json")))

    def test_wrong_number_sentence_removed(self):
        bad = json.loads(json.dumps(GOOD))
        bad["overall"].append("VIX는 77.7까지 치솟았다.")
        self.run_llm(caller_returning(json.dumps(bad)))
        out = json.load(open(os.path.join(self.tmp, "commentary.json"), encoding="utf-8"))
        self.assertEqual(len(out["overall"]), 2)
        self.assertNotIn("77.7", json.dumps(out, ensure_ascii=False))

    def test_tolerance_and_dates_ok(self):
        allowed = llm.allowed_numbers(llm.build_payload(DATA))
        self.assertTrue(llm.sentence_ok("HY OAS 4.3, 2026-10-09 기준 30일 변화.", allowed))   # 4.34 ±1%
        self.assertFalse(llm.sentence_ok("HY OAS 5.9.", allowed))

    def test_json_failure_retries_once_then_ok(self):
        c = caller_returning("not json", json.dumps(GOOD))
        self.assertEqual(self.run_llm(c), "generated")
        self.assertEqual(c.call_count, 2)

    def test_failure_keeps_previous_with_badge(self):
        self.run_llm(caller_returning(json.dumps(GOOD)))
        newer = dict(DATA, as_of="2099-01-01")
        c = caller_returning("x", "y", "z")
        self.assertEqual(llm.run(newer, self.tmp, caller=c), "failed")
        self.assertEqual(c.call_count, 3)
        out = json.load(open(os.path.join(self.tmp, "commentary.json"), encoding="utf-8"))
        self.assertTrue(out["update_failed"])
        self.assertEqual(out["as_of"], DATA["as_of"])      # 직전 해석 유지
        # 실패 표시된 캐시는 다음 빌드에서 재시도
        c2 = caller_returning(json.dumps(GOOD))
        self.assertEqual(llm.run(newer, self.tmp, caller=c2), "generated")


if __name__ == "__main__":
    unittest.main()
