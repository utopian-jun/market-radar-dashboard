"""6.5장 1년치 백필 (1회성) — 과거 값을 모아 레이더로그에 붙여넣을 CSV 하나를 만든다.

시트에는 직접 쓰지 않는다. 결과 CSV를 레이더로그 맨 아래에 붙여넣고 A열 오름차순 정렬.

  python backfill.py                       # 기존 수집일은 SHEET_CSV_URL(있으면)에서 읽어 제외
  python backfill.py --existing log.csv    # 로컬 CSV에서 제외할 날짜를 읽는다
  python backfill.py --days 365 --out out/radar_backfill.csv

출처: FRED fredgraph.csv(키 불필요) · yfinance(DXY·SPY·원달러).
공포탐욕(CNN)은 자동 수집이 막혀 있고, 풋콜(CBOE)은 공개 파일이 2019년에서 끊겨 있어 비워둔다.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, timedelta

import pandas as pd

from build import COLUMNS, SchemaError, load_csv_text, parse

FRED = {  # 열 → FRED 시리즈
    "HY OAS(%)": "BAMLH0A0HYM2",
    "VIX": "VIXCLS",
    "US10Y(%)": "DGS10",
    "WTI($)": "DCOILWTICO",
    "CCC OAS(%)": "BAMLH0A3HYC",
}
YF = {"DXY": "DX-Y.NYB", "SPY": "SPY", "원달러": "KRW=X", "CRWV": "CRWV", "ORCL": "ORCL"}
MA_WINDOW = 200
EMPTY_BY_DESIGN = ["공포탐욕", "공포탐욕 기준일", "풋콜 Total", "풋콜 Equity", "풋콜 Equity 5일평균", "CBOE 기준일",
                   "잠정판정", "확정판정", "추세", "신용", "변동성", "심리", "거시경고수", "AI신용"]


# ── 수집 (네트워크) ───────────────────────────────────────
def fetch_fred(series_id: str, start: date, end: date) -> pd.Series:
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", timeout=30,
                     params={"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")        # 휴장일은 '.' → NaN
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def fetch_yf(ticker: str, start: date, end: date) -> pd.Series:
    import yfinance as yf
    d = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                    progress=False, auto_adjust=False)
    if d is None or d.empty:
        raise RuntimeError("빈 결과")
    close = d["Close"]
    close = close.iloc[:, 0] if hasattr(close, "columns") else close
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.dropna()


# ── 조립 (순수 계산 — 테스트 대상) ────────────────────────
def assemble(series: dict[str, pd.Series], start: date, end: date, exclude: set[str]) -> pd.DataFrame:
    """열 이름 → 시계열 dict를 A~W 순서 DataFrame으로. 200일선은 SPY 전체 이력으로 계산."""
    spy = series.get("SPY")
    ma = spy.rolling(MA_WINDOW).mean() if spy is not None else None

    data = {c: s for c, s in series.items()}
    if ma is not None:
        data["SPY 200일선"] = ma
    frame = pd.DataFrame(data)
    frame = frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
    frame = frame[frame.index.dayofweek < 5]                         # 주말 제외
    frame = frame.dropna(how="all")                                   # 값이 하나도 없는 날(전 출처 휴장) 제외
    frame = frame[~frame.index.strftime("%Y-%m-%d").isin(exclude)]    # 이미 수집된 날짜는 제외
    frame = frame.sort_index()

    out = pd.DataFrame(index=frame.index, columns=COLUMNS, dtype=object)
    for c in frame.columns:
        out[c] = frame[c]
    out["수집일"] = frame.index.strftime("%Y-%m-%d")
    out["출처"] = "백필"
    out = out[COLUMNS]
    for c in ("HY OAS(%)", "US10Y(%)", "VIX", "WTI($)", "DXY", "SPY", "SPY 200일선", "원달러", "CCC OAS(%)", "CRWV", "ORCL"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out.reset_index(drop=True)


def existing_dates(source: str | None) -> set[str]:
    """레이더로그에 이미 있는 수집일. 읽지 못하면 빈 집합(호출부에서 경고)."""
    try:
        df = parse(load_csv_text(source))
    except (SchemaError, OSError) as e:
        print(f"경고: 기존 레이더로그를 못 읽었다 ({e})", file=sys.stderr)
        return set()
    return set(df["수집일"].dt.strftime("%Y-%m-%d"))


def summarize(out: pd.DataFrame) -> list[str]:
    lines = [f"행 {len(out)}개 · {out['수집일'].iloc[0]} ~ {out['수집일'].iloc[-1]}"]
    dup = out["수집일"].duplicated().sum()
    lines.append(f"날짜 오름차순 {'OK' if out['수집일'].is_monotonic_increasing else 'X'} · 중복 {dup}개 · 열 순서 {'OK' if list(out.columns) == COLUMNS else 'X'}")
    lines.append("결측 개수 (의도적으로 비운 열 제외):")
    for c in COLUMNS:
        if c in EMPTY_BY_DESIGN or c in ("수집일", "출처"):
            continue
        n = int(out[c].isna().sum())
        lines.append(f"  {c}: {n}{'  ← 전부 비어 있음' if n == len(out) else ''}")
    lines.append("의도적으로 비움: " + ", ".join(EMPTY_BY_DESIGN))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="오늘 기준 몇 일 전까지 (기본 365)")
    ap.add_argument("--existing", help="기존 레이더로그 CSV 경로/없으면 SHEET_CSV_URL")
    ap.add_argument("--out", default="out/radar_backfill.csv")
    args = ap.parse_args(argv)

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)
    exclude = existing_dates(args.existing) if (args.existing or os.environ.get("SHEET_CSV_URL")) else set()
    if not exclude:
        exclude = {date.today().isoformat()}
        print(f"경고: 기존 수집일을 몰라 오늘({date.today()})만 제외한다. 겹치는 날짜는 시트에서 수집 행이 이긴다", file=sys.stderr)
    else:
        print(f"기존 수집 {len(exclude)}일 제외")

    series: dict[str, pd.Series] = {}
    for col, sid in FRED.items():
        try:
            series[col] = fetch_fred(sid, start, end)
            print(f"FRED {sid:<14} → {col}: {len(series[col])}개")
        except Exception as e:      # noqa: BLE001
            print(f"경고: FRED {sid} 실패 → '{col}' 비움 ({str(e)[:120]})", file=sys.stderr)
    ma_start = start - timedelta(days=MA_WINDOW * 2)   # 200거래일 이동평균용 여분
    for col, tk in YF.items():
        try:
            series[col] = fetch_yf(tk, ma_start if col == "SPY" else start, end)
            print(f"yfinance {tk:<9} → {col}: {len(series[col])}개")
        except Exception as e:      # noqa: BLE001
            print(f"경고: yfinance {tk} 실패 → '{col}' 비움 ({str(e)[:120]})", file=sys.stderr)

    if not series:
        print("모든 출처가 실패했다", file=sys.stderr)
        return 1
    out = assemble(series, start, end, exclude)
    if out.empty:
        print("만들 행이 없다", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")      # 열 순서 = COLUMNS (A~W), 헤더 포함
    print(f"\n저장: {args.out}")
    print("\n".join(summarize(out)))
    print("\n시트 작업: ① W1에 '출처' 입력, 기존 수집 행의 W는 '수집'으로 채우기 "
          "② 이 CSV의 헤더를 뺀 행만 레이더로그 맨 아래에 붙여넣기 ③ 데이터 → 범위 정렬(A열 오름차순)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
