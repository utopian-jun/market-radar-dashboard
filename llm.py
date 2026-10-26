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
3. 지표별(indicators): 지금 어느 구간인지 + 30일 방향 + 수집 기간 평균 대비 위치를 1~2문장으로. 30일 전 값이 null이면 방향은 '데이터 부족'이라고만 쓴다.
4. 종합(overall): 층들(추세·신용·변동성·심리·거시경고)이 서로 맞물리는지/엇갈리는지(예: 심리 극공포인데 신용 평온), 경계선 근처 항목을 3~5문장으로.
5. watch: 판정이 바뀔 수 있는 조건 1~2개 (입력의 임계값 기준).
6. 한국어 반말, 간결하고 약간 냉소적인 톤. 약어는 첫 등장에 짧게 풀이(예: HY OAS는 정크본드 가산금리).
7. 수집 일수가 적으면(collection_days < 21) 평균·백분위는 참고용이라고 한 번 밝힌다.
JSON으로만 답한다."""

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
            "percentile", "thresholds", "zone", "near_threshold")}
    return {
        "as_of": data["as_of"],
        "collection_days": data["collection_days"],
        "verdict": data["verdict"],
        "layers": data["layers"],
        "indicators": inds,
        "near_threshold": data["near_threshold"],
        "latest_row": data["latest_row"],
    }


# ── 호출 ──────────────────────────────────────────────────
def call_gemini(payload: dict, model: str, api_key: str) -> str:
    import requests
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json",
                             "responseSchema": RESPONSE_SCHEMA},
    }
    r = requests.post(ENDPOINT.format(model=model), json=body, timeout=90,
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
