"""시장 레이더 대시보드 빌드 — 1단계: CSV → 검사 → 통계 → site/data.json

사용:
  python build.py --no-llm                 # SHEET_CSV_URL 환경변수의 CSV 사용
  python build.py --no-llm --csv sample/radar_log.csv
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

# ── 스키마 (레이더로그 A~V, 변경 금지) ─────────────────────────────
COLUMNS = [
    "수집일", "HY OAS(%)", "VIX", "US10Y(%)", "WTI($)", "DXY", "공포탐욕", "공포탐욕 기준일",
    "풋콜 Total", "풋콜 Equity", "풋콜 Equity 5일평균", "CBOE 기준일", "잠정판정", "확정판정",
    "추세", "신용", "변동성", "심리", "거시경고수", "SPY", "SPY 200일선", "원달러",
]
DATE_COLS = ["수집일", "공포탐욕 기준일", "CBOE 기준일"]
TEXT_COLS = ["잠정판정", "확정판정", "추세", "신용", "변동성", "심리"]
NUM_COLS = [c for c in COLUMNS if c not in DATE_COLS + TEXT_COLS]

# 지표 8개: 키 → (컬럼, 라벨, 임계선, 판정에 쓰는 컬럼)
# 임계값은 PLAN.md 5장 기준. 장기 평균/구간 설명은 5단계 이후 화면 콘텐츠에서 다룬다.
INDICATORS = {
    "HY_OAS":     {"col": "HY OAS(%)",           "label": "HY OAS",     "thresholds": [3.5, 5.0]},
    "VIX":        {"col": "VIX",                 "label": "VIX",        "thresholds": [15, 20, 30, 40]},
    "US10Y":      {"col": "US10Y(%)",            "label": "US10Y",      "thresholds": []},
    "WTI":        {"col": "WTI($)",              "label": "WTI",        "thresholds": []},
    "DXY":        {"col": "DXY",                 "label": "DXY",        "thresholds": []},
    "FEAR_GREED": {"col": "공포탐욕",             "label": "공포탐욕",    "thresholds": [25, 75]},
    "PC_TOTAL":   {"col": "풋콜 Total",           "label": "풋콜 Total",  "thresholds": [0.8, 1.1]},
    "PC_EQUITY":  {"col": "풋콜 Equity 5일평균",   "label": "풋콜 Equity", "thresholds": [0.55, 0.8]},
}
# 확정판정 → 현금 가이드 (시트 B30과 동일하게 유지). 빈 문자열이면 화면에서 가이드 줄을 숨긴다.
CASH_GUIDE = {
    "공격": "현금 5~10% · 추세 슬리브 한도 15%",
    "중립": "현금 ~20% · 슬리브 10%",
    "과열주의": "현금 15% · 신규 매수 속도 절반 · 슬리브 5%",
    "역발상기회": "현금 15~20% · 개시선 아래 종목 분할 매수 · 슬리브 0%",
    "방어": "현금 30%+ · 신규 금지 · 슬리브 0%",
}
NEAR_RATIO = 0.10  # 임계까지 10% 이내면 '경계선 근처'
STALE_DAYS = 3


class SchemaError(Exception):
    pass


def load_csv_text(csv_path: str | None) -> str:
    if csv_path:
        with open(csv_path, encoding="utf-8-sig") as f:
            return f.read()
    url = os.environ.get("SHEET_CSV_URL")
    if not url:
        raise SchemaError("CSV 소스가 없다: --csv 경로를 주거나 SHEET_CSV_URL 환경변수를 설정해라")
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def check_header(df: pd.DataFrame) -> None:
    got = [str(c).strip() for c in df.columns]
    if got == COLUMNS:
        return
    lines = []
    for i in range(max(len(got), len(COLUMNS))):
        exp = COLUMNS[i] if i < len(COLUMNS) else "(없음)"
        act = got[i] if i < len(got) else "(없음)"
        if exp != act:
            lines.append(f"  열 {_col_letter(i)}: 기대 '{exp}' / 실제 '{act}'")
    raise SchemaError(f"헤더 불일치 (기대 {len(COLUMNS)}개, 실제 {len(got)}개)\n" + "\n".join(lines))


def _col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    check_header(df)
    df.columns = COLUMNS
    df = df.apply(lambda s: s.str.strip())

    warns = 0
    for c in NUM_COLS:
        raw = df[c].str.replace(",", "", regex=False).replace("", pd.NA)
        num = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & num.isna()
        for idx in df.index[bad]:
            print(f"경고: {idx + 2}행 '{c}' 숫자 변환 실패 → null ({df.at[idx, c]!r})", file=sys.stderr)
            warns += 1
        df[c] = num
    for c in DATE_COLS:
        d = pd.to_datetime(df[c].replace("", pd.NA), errors="coerce")
        bad = df[c].ne("") & d.isna()
        for idx in df.index[bad]:
            print(f"경고: {idx + 2}행 '{c}' 날짜 변환 실패 → null ({df.at[idx, c]!r})", file=sys.stderr)
        df[c] = d
    for c in TEXT_COLS:
        df[c] = df[c].replace("", pd.NA)

    before = len(df)
    df = df[df["수집일"].notna()].sort_values("수집일").drop_duplicates("수집일", keep="last")
    if len(df) < before:
        print(f"경고: 수집일이 비었거나 중복인 {before - len(df)}행 제외", file=sys.stderr)
    if df.empty:
        raise SchemaError("유효한 데이터 행이 없다")
    return df.reset_index(drop=True)


# ── 통계 ──────────────────────────────────────────────────
def _f(v):
    """NaN/NA → None, numpy → python float."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), 4)


def _row_json(row: pd.Series) -> dict:
    out = {}
    for c in COLUMNS:
        v = row[c]
        if c in DATE_COLS:
            out[c] = None if pd.isna(v) else v.strftime("%Y-%m-%d")
        elif c in TEXT_COLS:
            out[c] = None if pd.isna(v) else str(v)
        else:
            out[c] = _f(v)
    return out


def near_threshold(value, thresholds):
    """가장 가까운 임계값과 상대거리. 10% 이내면 dict, 아니면 None."""
    if value is None or not thresholds:
        return None
    t = min(thresholds, key=lambda x: abs(value - x))
    dist = abs(value - t) / abs(t) if t else None
    if dist is not None and dist <= NEAR_RATIO:
        return {"threshold": t, "distance_pct": round(dist * 100, 1)}
    return None


def zone(value, thresholds):
    """임계값 기준 구간 번호 (0 = 가장 낮은 구간). 임계 없으면 None."""
    if value is None or not thresholds:
        return None
    return sum(1 for t in thresholds if value >= t)


def indicator_stats(df: pd.DataFrame, spec: dict, latest_date: pd.Timestamp) -> dict:
    s = df[spec["col"]]
    valid = s.dropna()
    cur = _f(s.iloc[-1])
    # 30일 전: 최신일 - 30일 이하 중 가장 가까운 유효 행
    past = df[(df["수집일"] <= latest_date - pd.Timedelta(days=30)) & s.notna()]
    prev = _f(past[spec["col"]].iloc[-1]) if not past.empty else None
    prev_date = past["수집일"].iloc[-1].strftime("%Y-%m-%d") if not past.empty else None
    change = None if (cur is None or prev is None) else round(cur - prev, 4)
    change_pct = None if (change is None or not prev) else round(change / abs(prev) * 100, 2)
    pct_rank = None
    if cur is not None and len(valid) >= 2:
        pct_rank = round(float((valid <= cur).mean() * 100), 1)
    return {
        "label": spec["label"],
        "column": spec["col"],
        "current": cur,
        "prev_30d": prev,
        "prev_30d_date": prev_date,
        "change_30d": change,
        "change_30d_pct": change_pct,
        "mean": _f(valid.mean()) if len(valid) else None,
        "min": _f(valid.min()) if len(valid) else None,
        "max": _f(valid.max()) if len(valid) else None,
        "percentile": pct_rank,          # 수집 기간 내 백분위 (0~100)
        "n": int(len(valid)),
        "thresholds": spec["thresholds"],
        "zone": zone(cur, spec["thresholds"]),
        "near_threshold": near_threshold(cur, spec["thresholds"]),
    }


def provisional_streak(df: pd.DataFrame) -> int | None:
    """잠정판정이 최신 값 그대로 이어진 연속 일수."""
    s = df["잠정판정"]
    if s.isna().iloc[-1]:
        return None
    last, n = s.iloc[-1], 0
    for v in reversed(s.tolist()):
        if v == last:
            n += 1
        else:
            break
    return n


def build_data(df: pd.DataFrame, today: date | None = None) -> dict:
    today = today or datetime.now(timezone.utc).astimezone().date()
    latest = df.iloc[-1]
    latest_date = df["수집일"].iloc[-1]
    age = (pd.Timestamp(today) - latest_date).days

    indicators = {k: indicator_stats(df, spec, latest_date) for k, spec in INDICATORS.items()}
    prov, conf = latest["잠정판정"], latest["확정판정"]

    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in df["수집일"]],
        **{k: [_f(v) for v in df[spec["col"]]] for k, spec in INDICATORS.items()},
        "SPY": [_f(v) for v in df["SPY"]],
        "SPY_200MA": [_f(v) for v in df["SPY 200일선"]],
        "USDKRW": [_f(v) for v in df["원달러"]],
        "verdict": [None if pd.isna(v) else str(v) for v in df["확정판정"]],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": latest_date.strftime("%Y-%m-%d"),
        "days_since_as_of": int(age),
        "stale": bool(age > STALE_DAYS),
        "collection_days": int(len(df)),   # 수집 N일차 표기용
        "first_date": df["수집일"].iloc[0].strftime("%Y-%m-%d"),
        "verdict": {
            "confirmed": None if pd.isna(conf) else str(conf),
            "provisional": None if pd.isna(prov) else str(prov),
            "cash_guide": CASH_GUIDE.get(str(conf), "") or None,
            "differs": bool(pd.notna(prov) and pd.notna(conf) and prov != conf),
            "provisional_streak_days": provisional_streak(df),
        },
        "layers": {
            "trend": None if pd.isna(latest["추세"]) else str(latest["추세"]),
            "credit": None if pd.isna(latest["신용"]) else str(latest["신용"]),
            "volatility": None if pd.isna(latest["변동성"]) else str(latest["변동성"]),
            "sentiment": None if pd.isna(latest["심리"]) else str(latest["심리"]),
            "macro_warnings": _f(latest["거시경고수"]),
        },
        "indicators": indicators,
        "near_threshold": [k for k, v in indicators.items() if v["near_threshold"]],
        "latest_row": _row_json(latest),
        "series": series,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="로컬 CSV 경로 (없으면 SHEET_CSV_URL 사용)")
    ap.add_argument("--out", default="site", help="출력 디렉터리")
    ap.add_argument("--no-llm", action="store_true", help="Gemini API 호출 건너뜀")
    ap.add_argument("--force-llm", action="store_true", help="캐시가 있어도 해석을 다시 생성")
    args = ap.parse_args(argv)

    try:
        df = parse(load_csv_text(args.csv))
    except SchemaError as e:
        print(f"빌드 실패: {e}", file=sys.stderr)
        return 1

    data = build_data(df)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"data.json 생성: {path} (행 {len(df)}개, 기준일 {data['as_of']}, 수집 {data['collection_days']}일차)")
    if data["stale"]:
        print(f"경고: 최신 데이터가 {data['days_since_as_of']}일 전이다", file=sys.stderr)
    if not args.no_llm:
        import llm
        llm.run(data, args.out, force=args.force_llm)   # 실패해도 빌드는 성공 (직전 해석 유지)
    return 0


if __name__ == "__main__":
    sys.exit(main())
