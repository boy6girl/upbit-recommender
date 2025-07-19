import os
import time
import uuid
import json
import threading
import requests
from collections import deque, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyupbit
import websocket
from dotenv import load_dotenv

# ─── 설정 ─────────────────────────────────────────────────────────────
load_dotenv()
TG_TOKEN        = os.getenv("TELEGRAM_TOKEN")
CHAT_ID         = os.getenv("CHAT_ID")
TAKE_PROFIT_PCT = 1.0    # 익절 목표 +1%
STOP_LOSS_PCT   = 0.5    # 손절 한계 -0.5%

# ─── 헬퍼 함수 ────────────────────────────────────────────────────────
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    except:
        pass

def compute_rsi(arr, period=14):
    s = pd.Series(arr, dtype=float)
    delta = s.diff().dropna()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 0

def compute_macd_hist(arr):
    s = pd.Series(arr, dtype=float)
    macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    return float((macd - signal).iloc[-1]) if not macd.empty and not signal.empty else 0

def compute_vwap(df):
    vol_sum = df['volume'].sum()
    return float((df['close'] * df['volume']).sum() / vol_sum) if vol_sum > 0 else 0

# ─── 가격 수집 ───────────────────────────────────────────────────────
CANDIDATES = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices = {}
price_history = defaultdict(lambda: deque(maxlen=12))

def seed_initial_prices():
    for code in CANDIDATES:
        try:
            price = pyupbit.get_current_price(code)
            if isinstance(price, dict):
                price = price.get(code)
            if price and price > 0:
                latest_prices[code] = price
                for _ in range(3):
                    price_history[code].append((time.time(), price))
        except:
            continue

def on_message(ws, msg):
    data = json.loads(msg)
    code = data.get("code")
    price = data.get("trade_price")
    if code and isinstance(price, (int, float)):
        latest_prices[code] = price
        price_history[code].append((time.time(), price))

def on_open(ws):
    ws.send(json.dumps([{"ticket": str(uuid.uuid4())},
                        {"type": "ticker", "codes": CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

# ─── 점수 & 확률 계산 ────────────────────────────────────────────────
def backtest_probabilities(codes, samples=100, period=14):
    results = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "minute5", count=samples + period + 1)
        if df is None or len(df) < samples + period + 1:
            continue
        closes = df['close'].values
        for i in range(samples):
            window = closes[i:i+period+1]
            rsi = compute_rsi(window, period)
            score = 100 - rsi
            hit = (closes[i+period] - closes[i+period-1]) > 0
            results.append((score, hit))
    if not results:
        return {b: 0.6 for b in range(0, 101, 10)}
    df = pd.DataFrame(results, columns=["score", "hit"])
    df["bucket"] = (df["score"] // 10) * 10
    prob = df.groupby("bucket")["hit"].mean().to_dict()
    return {b: prob.get(b, 0.0) for b in range(0, 101, 10)}

def recommend(prob_map, top_n=3):
    scored = []
    for code, price in latest_prices.items():
        df = pyupbit.get_ohlcv(code, "minute5", count=20)
        if df is None or len(df) < 20:
            continue
        rsi = compute_rsi(df['close'].values)
        macd = compute_macd_hist(df['close'].values)
        vwap = compute_vwap(df)
        score = (100 - rsi) * 0.5 + max(macd, 0) * 0.3 + max((df['close'].iloc[-1] - vwap) / vwap * 100, 0) * 0.2
        if np.isnan(score):
            continue
        bucket = int(score // 10) * 10
        prob = prob_map.get(bucket, 0.6)
        scored.append((code, price, score, prob))

    best = sorted(scored, key=lambda x: x[2], reverse=True)[:top_n]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"{now} 추천 종목 (Top {top_n})"]
    for i, (code, price, score, prob) in enumerate(best, 1):
        tp = price * (1 + TAKE_PROFIT_PCT / 100)
        sl = price * (1 - STOP_LOSS_PCT / 100)
        lines.append(f"{i}. {code}\n점수: {score:.1f} / 확률: {prob*100:.1f}%\n익절: {tp:.0f}원 / 손절: {sl:.0f}원\n")

    result = "\n".join(lines)
    print(result)
    for _, _, score, prob in best:
        if (score >= 80 and prob >= 0.7) or score >= 90 or prob >= 0.9:
            send_telegram(result)
            break

# ─── 메인 실행 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    seed_initial_prices()
    prob_map = backtest_probabilities(CANDIDATES)
    threading.Thread(target=start_ws, daemon=True).start()
    time.sleep(30)

    START_TIME = datetime.now()  # ⏱️ 시작 시간 저장

    while True:
        recommend(prob_map, top_n=3)

        # ⏰ 6시간 지나면 종료
        if datetime.now() - START_TIME > timedelta(hours=6):
            print("✅ 6시간 경과. 자동 종료합니다.")
            break

        time.sleep(60)  # 1분 주기 실행
