import os
import time
import uuid
import json
import threading
import requests
import math
from collections import deque, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyupbit
import websocket
from dotenv import load_dotenv

# ─── 0) 환경 변수 & 목표 퍼센트 ───────────────────────────────────────
load_dotenv()
TG_TOKEN        = os.getenv("TELEGRAM_TOKEN")
CHAT_ID         = os.getenv("CHAT_ID")
TAKE_PROFIT_PCT = 1.0   # 익절 목표 +1%
STOP_LOSS_PCT   = 0.5   # 손절 한계 -0.5%

def send_telegram(msg: str):
    """텔레그램 메시지 전송"""
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    except Exception:
        pass

# ─── 1) 기본 설정 ────────────────────────────────────────────────────
CANDIDATES     = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices  = {}
price_history  = defaultdict(lambda: deque(maxlen=12))

# ─── 2) 초기 REST 시드 ─────────────────────────────────────────────────
def seed_initial_prices():
    print("▶️ 초기 가격 시드 시작")
    for code in CANDIDATES:
        price = None
        try:
            r = pyupbit.get_current_price(code)
            if isinstance(r, (int, float)):
                price = r
            elif isinstance(r, dict):
                price = r.get("trade_price")
            elif isinstance(r, list) and r:
                price = r[0].get("trade_price")
        except:
            pass
        if not price:
            try:
                r = requests.get(
                    "https://api.upbit.com/v1/ticker",
                    params={"markets": code}, timeout=5
                ).json()
                price = r[0].get("trade_price")
            except:
                pass
        if price and price > 0:
            latest_prices[code] = price
            for _ in range(3):
                price_history[code].append((time.time(), price))
    print(f"✅ 초기 가격 시드 완료: {len(latest_prices)} 종목")

# ─── 3) 5분봉 백테스트 & 확률 매핑 ────────────────────────────────────
def compute_rsi(arr, period=14):
    s = pd.Series(arr, dtype=float)
    d = s.diff().dropna()
    up   = d.clip(lower=0).rolling(period).mean()
    down = -d.clip(upper=0).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    return float(rsi.fillna(100).iloc[-1])

def backtest_probabilities(codes, samples=100, period=14):
    print("▶️ 5분봉 백테스트 시작")
    results = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "minute5", count=samples+period+1)
        if df is None or len(df) < samples+period+1:
            continue
        closes = df['close'].values
        for i in range(samples):
            window = closes[i:i+period+1]
            rsi = compute_rsi(window, period)
            score = 100 - rsi
            hit = (closes[i+period] - closes[i+period-1]) > 0
            results.append((score, hit))
    if not results:
        pm = {b: 0.6 for b in range(0, 101, 10)}
    else:
        df = pd.DataFrame(results, columns=["score","hit"])
        df["bucket"] = (df["score"] // 10) * 10
        prob = df.groupby("bucket")["hit"].mean().to_dict()
        pm = {b: prob.get(b, 0.0) for b in range(0, 101, 10)}
    print("✅ 5분봉 백테스트 완료:", pm)
    return pm

# ─── 4) 5분봉 지표 및 추천 ─────────────────────────────────────────────
def compute_macd_hist(arr):
    s = pd.Series(arr, dtype=float)
    macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig  = macd.ewm(span=9).mean()
    val  = (macd - sig).iloc[-1]
    return float(pd.Series([val]).fillna(0)[0])

def compute_vwap(df):
    vol = df['volume'].sum()
    return float((df['close'] * df['volume']).sum() / vol) if vol>0 else 0.0

def collect_and_score_5m(prob_map, strict=True):
    now = datetime.now()
    scored = []
    for code, price in latest_prices.items():
        df = pyupbit.get_ohlcv(code, "minute5", count=20)
        if df is None or len(df) < 20:
            continue
        if strict and (now - df.index[0].to_pydatetime() < timedelta(days=7)):
            continue
        if strict and df['volume'].iloc[-1] < df['volume'].mean():
            continue

        arr   = df['close'].values
        rsi   = compute_rsi(arr)
        macd  = compute_macd_hist(arr)
        vwap  = compute_vwap(df)
        score = (100 - rsi)*0.5 + max(macd,0)*0.3 + max((df['close'].iloc[-1] - vwap)/vwap*100,0)*0.2
        if math.isnan(score):
            continue

        bucket = int(score//10)*10
        prob   = prob_map.get(bucket, 0.6)
        scored.append((code, price, score, prob))
    return scored

def recommend_5m(prob_map, top_n=3):
    print("▶️ 5분봉 추천 시작")
    cands = collect_and_score_5m(prob_map, strict=True)
    if not cands:
        cands = collect_and_score_5m(prob_map, strict=False)
    best = sorted(cands, key=lambda x: x[2], reverse=True)[:top_n]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = [f"{now} 🔔 5m Top{top_n}:"]
    for i,(c,p,sc,pr) in enumerate(best,1):
        tp = p*(1+TAKE_PROFIT_PCT/100)
        sl = p*(1-STOP_LOSS_PCT/100)
        msg.append(f"{i}. {c} | 종가:{p:,.0f}원 | 점수:{sc:.1f} | 확률:{pr*100:.1f}% | 익절:{tp:,.0f}원 | 손절:{sl:,.0f}원")
    full_msg = "\n".join(msg)
    print(full_msg)                  # 터미널에 추천 리스트 출력
    send_telegram(full_msg)
    print("✅ 5분봉 추천 완료")

# ─── 5) 일봉 백테스트 & 확률 매핑 ────────────────────────────────────
SCORE_WINDOW = 30

def compute_daily_score(df):
    close = df['close']
    rsi   = compute_rsi(close.values)
    ema20 = close.ewm(span=20).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]
    adx   = ((df['high'] - df['low']) / df['close']).rolling(14).mean().iloc[-1] * 100
    bb_w  = ((close.rolling(20).mean() + 2*close.rolling(20).std()) -
             (close.rolling(20).mean() - 2*close.rolling(20).std())).iloc[-1] \
            / close.rolling(20).mean().iloc[-1] * 100
    today = df.iloc[-1]
    today_pct = max((today['close'] - today['open'])/today['open']*100, 0)
    vol_score = df['volume'].iloc[-1] / df['volume'].tail(20).mean() * 100
    score = ((100 - rsi)*0.2 +
             max((ema20-ema50)/ema50*100,0)*0.15 +
             adx*0.1 + bb_w*0.1 +
             today_pct*0.15 + vol_score*0.15)
    return score

def backtest_daily_probabilities(codes, lookback=250):
    print("▶️ 일봉 백테스트 시작")
    results = []
    count = lookback + SCORE_WINDOW + 1
    for code in codes:
        df = pyupbit.get_ohlcv(code, "day", count=count)
        if df is None or len(df) < count:
            continue
        for i in range(lookback):
            window = df.iloc[i : i+SCORE_WINDOW]
            today  = window.iloc[-1]
            nxt    = df.iloc[i+SCORE_WINDOW]
            score  = compute_daily_score(window)
            bucket = int(score//10)*10
            hit    = nxt['close'] > today['close']
            results.append((bucket, hit))
    if not results:
        pm = {b: 0.6 for b in range(0, 101, 10)}
    else:
        dfp = pd.DataFrame(results, columns=["bucket","hit"])
        prob = dfp.groupby("bucket")["hit"].mean().to_dict()
        pm = {b: prob.get(b, 0.0) for b in range(0, 101, 10)}
    print("✅ 일봉 백테스트 완료:", pm)
    return pm

def daily_recommend(codes, prob_map_day, top_n=3):
    print("▶️ 일봉 추천 시작")
    scored = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "day", count=SCORE_WINDOW)
        if df is None or df.empty:
            continue
        price  = df['close'].iloc[-1]
        score  = compute_daily_score(df)
        bucket = int(score//10)*10
        prob   = prob_map_day.get(bucket, 0.0)
        scored.append((code, price, score, prob))
    best = sorted(scored, key=lambda x: x[2], reverse=True)[:top_n]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = [f"{now} 📈 Daily Top{top_n}:"]
    for i,(c,p,sc,pr) in enumerate(best,1):
        tp = p*(1+TAKE_PROFIT_PCT/100)
        sl = p*(1-STOP_LOSS_PCT/100)
        msg.append(f"{i}. {c} | 종가:{p:,.0f}원 | 점수:{sc:.1f} | 확률:{pr*100:.1f}% | 익절:{tp:,.0f}원 | 손절:{sl:,.0f}원")
    full_msg = "\n".join(msg)
    print(full_msg)                  # 터미널에 추천 리스트 출력
    send_telegram(full_msg)
    print("✅ 일봉 추천 완료")

# ─── 6) WebSocket 콜백 & 실행 진입점 ─────────────────────────────────
def on_message(ws, msg):
    d, p = json.loads(msg).get("code"), json.loads(msg).get("trade_price")
    if d and isinstance(p, (int, float)):
        latest_prices[d] = p
        price_history[d].append((time.time(), p))

def on_open(ws):
    ws.send(json.dumps([{"ticket": str(uuid.uuid4())},
                        {"type": "ticker", "codes": CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_open=on_open, on_message=on_message
    )
    ws.run_forever()

if __name__ == "__main__":
    print("▶️ recommender_ws.py 시작")
    seed_initial_prices()
    prob_map_5m  = backtest_probabilities(CANDIDATES, samples=100, period=14)
    prob_map_day = backtest_daily_probabilities(CANDIDATES, lookback=250)

    threading.Thread(target=start_ws, daemon=True).start()
    print("✅ WebSocket 스레드 시작, 30초 대기…")
    time.sleep(30)

    daily_sent = False
    while True:
        try:
            recommend_5m(prob_map_5m, top_n=3)

            now = datetime.now()
            if now.hour == 9 and now.minute == 5 and not daily_sent:
                daily_recommend(CANDIDATES, prob_map_day, top_n=3)
                daily_sent = True
            if now.hour == 9 and now.minute == 6:
                daily_sent = False

            time.sleep(60)
        except Exception as e:
            print(f"[메인 루프 에러] {e}")
            time.sleep(30)
