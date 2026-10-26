"""4단계 — Gemini API 해석 생성: 호출 · 날짜별 캐시 · 숫자 검증 · 실패 대비

키는 환경변수 GEMINI_API_KEY (GitHub Secrets). 모델은 LLM_MODEL (기본 gemini-3.5-flash-lite).
출력: <out>/commentary.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime

DEFAULT_MODEL = "gemini-3.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEYS = ["HY_OAS", "VIX", "US10Y", "WTI", "DXY", "FEAR_GREED", "PC_TOTAL", "PC_EQUITY"]
TOLERANCE = 0.01          # 입력값 대비 ±1%
CONTEXT_NUMBERS = {1, 2, 3, 4, 5, 10, 30, 200}   # '30일', '5일 평균' 같은 문맥 숫자

SYSTEM_PROMPT = """너는 미국 시장 지표 대시보드의 해석 담당이다. 규칙:
1. 제공된 JSON의 숫자만 근거로 쓴다. 뉴스·이벤트·원인 추측 금지. 새 숫자를 계산해서 만들지 말고 주어진 숫자를 그대로 인용한다.
2. 매수·매도·비중 지시 금지. 판정과 현금 가이드는 주어진 값을 인용만 한다.
3. 지표별(indicators): 2~3문장. 숫자 나열이 아니라 의미를 말한다 — (1) 지금 구간 (2) 30일 변화의 크기가 guide의 기준(속도·경보 수준)에 비춰 큰지 작은지 (3) percentile로 기간 중 어느 정도 위치인지(예: 상위 1%) (4) 그래서 무엇을 시사하는지(guide 근거). 변화가 크면 그 점을 먼저 말한다. 평균은 '1년 평균'이 아니라 '기간 평균'이라 부르고 '수집 기간'이라는 말은 쓰지 않는다. 30일 전 값이 null이면 '30일 전 비교 값이 없다'고 짧게 말한다(이 지시문을 따라 쓰지 않는다).
4. 종합(overall): 4~6문장. 첫 문장은 오늘의 핵심 한 줄. 층들(추세·신용·변동성·심리·거시경고)이 맞물리는지/엇갈리는지와 그 의미, 30일 변화가 가장 큰 지표 2~3개, 경계선 근처 항목을 말한다. 어떤 지표가 guide의 경보 기준을 넘었는데 층 판정(예: 거시경고수)에 안 잡혀 있으면 '기준을 넘었다'고 사실만 짚는다(판정이 틀렸다고 단정하지 않는다).
5. watch: 판정이 바뀔 수 있는 조건 1~2개 (입력의 임계값 기준).
6. 지표는 label(예: HY OAS, 공포탐욕, 풋콜 Equity)로 부르고 JSON 키 이름(FEAR_GREED 등)은 쓰지 않는다. 구간은 zone_label 이름으로 말하고 구간 번호는 절대 쓰지 않는다(zone_label이 null이면 구간을 말하지 않는다). 숫자는 입력에 적힌 그대로 쓴다.
7. 한국어 반말, 간결하고 약간 냉소적인 톤. 약어는 첫 등장에 짧게 풀이(예: HY OAS는 정크본드 가산금리).
8. 현금 가이드 문구는 화면 위쪽 카드에 이미 있으니 종합 해석에서 되풀이하지 않는다. 판정 이름은 언급해도 된다.
10. ccc_oas가 있으면 종합에서 HY OAS와 비교해 약한 고리가 벌어지는지 한 문장 이상 말한다(gap_vs_hy는 'CCC−HY 갭'이라고 부르고 키 이름은 쓰지 않는다). ai_credit이 있으면 종합에서 한 문장으로 다루되, 시장 전체 판정에는 반영되지 않는 AI 섹터 경보임을 밝히고 기준일(as_of)이 오늘과 다르면 그 날짜를 말한다. 이 둘은 indicators가 아니라 overall에서만 다룬다.
11. 평균·최소·최대·백분위·30일 전 값은 수집+백필 전체(history_days) 기준이다. history_days가 21 미만이면 평균·백분위는 참고용이라고 한 번 밝힌다. 백필 구간에는 판정이 없으니 판정 이력을 말하지 않는다.
JSON으로만 답한다."""

GUIDE = {  # PLAN.md 5장 해석 기준 요약
    "HY_OAS": "정크본드 금리-국채 금리. 3.5% 미만 평온(지나치면 안일), 3.5~5% 주의, 5% 초과 스트레스. 수준보다 속도가 중요: 한 달 +1%p면 경보. 주식보다 먼저 움직이는 경향.",
    "VIX": "S&P500 30일 예상 변동성. 15 미만 안일, 20~30 불안, 30 초과 공포, 40 초과 패닉. 급등 뒤 꺾이는 시점이 단기 바닥인 경우가 많다.",
    "US10Y": "10년 국채 금리, 모든 자산의 할인율. 오르면 고PER 기술주에 불리. 수준보다 한 달 +0.4%p 같은 급등이 문제.",
    "WTI": "원유 가격. 급등은 인플레 재점화·금리 인하 지연, 급락은 수요 둔화 신호일 수 있다. 거시 경고 기준은 한 달 +15%.",
    "DXY": "달러 가치. 강세는 글로벌 유동성 긴축·미국 다국적기업 실적 역풍, 원화 투자자에겐 달러 매수 단가 상승.",
    "FEAR_GREED": "반대로 읽는 지표. 25 이하 극공포는 역발상 기회 후보(단 신용이 멀쩡할 때만), 75 이상 극탐욕은 과열 경계.",
    "PC_TOTAL": "전체 옵션 풋/콜. 1.1 이상 공포, 0.8 이하 낙관. 참고용.",
    "PC_EQUITY": "개별주 옵션 풋/콜 5일 평균, 개인 심리에 가까움. 반대로 읽기: 0.8 이상 공포(역발상), 0.55 이하 탐욕.",
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall": {"type": "ARRAY", "items": {"type": "STRING"}},
        "indicators": {"type": "OBJECT", "properties": {k: {"type": "STRING"} for k in KEYS}, "required": KEYS},
        "watch": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["overall", "indicators", "watch"],
}


# ── 입력 구성 ─────────────────────────────────────────────
def build_payload(data: dict) -> dict:
    inds = {}
    for k, s in data["indicators"].items():
        inds[k] = {key: s[key] for key in (
            "label", "current", "prev_30d", "change_30d", "change_30d_pct", "mean", "min", "max",
            "percentile", "thresholds", "zone_label", "near_threshold")}
        inds[k]["guide"] = GUIDE.get(k)
        for key in ("current", "prev_30d", "change_30d", "mean", "min", "max"):
            if isinstance(inds[k][key], float):
                inds[k][key] = round(inds[k][key], 2)     # 모델이 소수 4자리를 그대로 읊지 않게
    return {
        "as_of": data["as_of"],
        "collection_days": data["collection_days"],
        "backfill_days": data.get("backfill_days", 0),
        "history_days": data.get("history_days", data["collection_days"]),
        "verdict": data["verdict"],
        "layers": data["layers"],
        "indicators": inds,
        "near_threshold": data["near_threshold"],
        "latest_row": data["latest_row"],
        "ccc_oas": _ccc(data),
        "ai_credit": _ai(data),
    }


def _ccc(data: dict):
    c = (data.get("extras") or {}).get("CCC")
    if not c or c.get("current") is None:
        return None
    r = lambda v: None if v is None else round(v, 2)
    return {"label": "CCC OAS(%)", "current": r(c["current"]), "as_of": c.get("as_of"), "change_30d": r(c.get("change_30d")),
            "gap_vs_hy": r(c.get("gap_vs_hy")), "percentile": c.get("percentile"), "mean": r(c.get("mean")),
            "guide": "CCC 등급 회사채 스프레드(HY 중 가장 약한 고리). HY OAS보다 먼저·더 크게 벌어지는지 본다. 신용 판정 기준: CCC 30일 +1.0%p면 주의, +2.0%p면 경보."}


def _ai(data: dict):
    a = data.get("ai_credit")
    if not a or a.get("verdict") is None:
        return None
    return {"verdict": a["verdict"], "crwv_1m_pct": a.get("crwv_1m_pct"), "orcl_1m_pct": a.get("orcl_1m_pct"), "as_of": a.get("as_of"),
            "rules": a.get("rules"), "guide": "코어위브(CRWV)·오라클(ORCL) 1개월 주가 변화율로 본 AI 섹터 신용 경보. 시장 전체 판정(현금 가이드)에는 반영되지 않는 별도 섹터 신호."}


# ── 호출 ──────────────────────────────────────────────────
def call_gemini(payload: dict, model: str, api_key: str) -> str:
    import requests
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json",
                             "responseSchema": RESPONSE_SCHEMA},
    }
    r = requests.post(ENDPOINT.format(model=model), json=body, timeout=150,
                      headers={"x-goog-api-key": api_key})   # 키는 헤더로만 (URL·로그에 안 남김)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


# ── 검증 ──────────────────────────────────────────────────
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _flatten_numbers(obj, out: set) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(_DATE.sub(" ", obj)):
            try:
                out.add(abs(float(m.replace(",", ""))))
            except ValueError:
                pass


def allowed_numbers(payload: dict) -> set:
    nums: set = set()
    _flatten_numbers(payload, nums)
    return nums


def _num_ok(x: float, allowed: set) -> bool:
    x = abs(x)
    if x in CONTEXT_NUMBERS:
        return True
    return any(abs(x - a) <= max(TOLERANCE * a, 0.005) for a in allowed)


def sentence_ok(text: str, allowed: set) -> bool:
    for m in _NUM.findall(_DATE.sub(" ", text)):
        try:
            if not _num_ok(float(m.replace(",", "")), allowed):
                return False
        except ValueError:
            continue
    return True


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def validate(out: dict, allowed: set) -> tuple[dict, list[str]]:
    """입력에 없는 숫자(±1% 밖)가 든 문장을 제거한다. 구조가 틀리면 ValueError."""
    if not isinstance(out, dict) or not isinstance(out.get("overall"), list) \
            or not isinstance(out.get("indicators"), dict) or not isinstance(out.get("watch", []), list):
        raise ValueError("해석 JSON 구조가 틀림")
    warns: list[str] = []

    def clean(text, where):
        kept = []
        for s in _sentences(str(text)):
            if sentence_ok(s, allowed):
                kept.append(s)
            else:
                warns.append(f"{where}: 숫자 불일치로 문장 제거 → {s}")
        return kept

    overall = [s for t in out["overall"] for s in clean(t, "overall")]
    watch = [s for t in out.get("watch", []) for s in clean(t, "watch")]
    indicators = {k: " ".join(clean(out["indicators"].get(k, ""), k)) for k in KEYS}
    if not overall:
        raise ValueError("검증 후 종합 해석이 비었다")
    return {"overall": overall, "indicators": indicators, "watch": watch}, warns


# ── 캐시 + 실행 ───────────────────────────────────────────
def _load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run(data: dict, out_dir: str, *, force: bool = False, caller=call_gemini) -> str:
    """반환: 'skipped-no-key' | 'cached' | 'generated' | 'failed'"""
    path = os.path.join(out_dir, "commentary.json")
    prev = _load(path)
    if prev and prev.get("as_of") == data["as_of"] and not prev.get("update_failed") and not force:
        print(f"해석 캐시 사용 ({data['as_of']}) — API 호출 없음")
        return "cached"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY 없음 — 해석 단계 건너뜀 (나머지는 정상)", file=sys.stderr)
        return "skipped-no-key"

    model = os.environ.get("LLM_MODEL") or DEFAULT_MODEL   # 빈 문자열(미설정 변수)도 기본값으로
    payload = build_payload(data)
    allowed = allowed_numbers(payload)
    result, last_err = None, None
    for attempt in (1, 2):          # JSON/호출 실패 시 1회 재시도
        try:
            result, warns = validate(json.loads(caller(payload, model, api_key)), allowed)
            for w in warns:
                print("경고:", w, file=sys.stderr)
            break
        except Exception as e:      # noqa: BLE001 — 어떤 실패든 직전 해석 유지로 처리
            last_err = e
            print(f"해석 생성 실패 ({attempt}/2): {str(e)[:200]}", file=sys.stderr)

    if result is None:
        if prev:                    # 직전 해석 유지 + 실패 배지
            prev["update_failed"] = True
            _write(path, prev)
        print("직전 해석을 유지한다" if prev else "직전 해석도 없어 해석 칸을 비워둔다", file=sys.stderr)
        return "failed"

    _write(path, {"as_of": data["as_of"], "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                  "model": model, "update_failed": False, **result})
    print(f"commentary.json 생성: {path} ({model})")
    return "generated"


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
