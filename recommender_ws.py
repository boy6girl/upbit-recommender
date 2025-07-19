import os
import time, uuid, json
import requests
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
TAKE_PROFIT_PCT = 1.0   # 익절 목표: +1.0%
STOP_LOSS_PCT   = 0.5   # 손절 한계: -0.5%

# ─── 1) 공통 지표 계산 함수 ─────────────────────────────────────────────
def compute_rsi(closes, period=14):
    s     = pd.Series(closes)
    delta = s.diff().dropna()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    rs    = up / down.replace(0, np.nan)
    return (100 - 100/(1 + rs)).iloc[-1]

def compute_macd_hist(closes):
    s      = pd.Series(closes)
    macd   = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal).iloc[-1]

def compute_momentum(closes):
    if len(closes) < 2:
        return 0.0
    p0, p1 = closes[0], closes[-1]
    return ((p1 - p0) / p0) * 100

# ─── 2) 공통 설정 ────────────────────────────────────────────────────
CANDIDATES    = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices = {}
price_history = defaultdict(lambda: deque(maxlen=12))

# ─── 3) 초기 가격 시드 ────────────────────────────────────────────────
def seed_initial_prices():
    for code in CANDIDATES:
        try:
            price = pyupbit.get_current_price(code)
            if price:
                latest_prices[code] = price
                for _ in range(3):
                    price_history[code].append((time.time(), price))
        except:
            continue

# ─── 4) WebSocket 실시간 수집 ─────────────────────────────────────────
def on_message(ws, msg):
    data  = json.loads(msg)
    code  = data.get("code")
    price = data.get("trade_price")
    if code and price is not None:
        latest_prices[code] = price
        price_history[code].append((time.time(), price))

def on_open(ws):
    ws.send(json.dumps([{"ticket":str(uuid.uuid4())}, {"type":"ticker","codes":CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp("wss://api.upbit.com/websocket/v1", on_message=on_message, on_open=on_open)
    ws.run_forever()

# ─── 5) 알림 함수 ─────────────────────────────────────────────────────
def send_telegram(message: str):
    url     = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except:
        pass

# ─── 6) 후보 수집 (5분봉) ─────────────────────────────────────────────
def collect_5min_candidates(strict, prob_map):
    candidates = []
    now        = datetime.now()
    for code, price in latest_prices.items():
        try:
            df = pyupbit.get_ohlcv(code, "minute5", count=15)
            if df is None or df.empty:
                continue
            if strict and (now - df.index[0] < timedelta(days=7)):
                continue
            closes = df['close'].values; vols = df['volume'].values
            if strict and (vols[-1] < vols[:-1].mean()):
                continue
            rsi   = compute_rsi(closes)
            macd  = compute_macd_hist(closes)
            mom   = compute_momentum([p for _,p in price_history[code]])
            score = (100 - rsi)*0.5 + max(macd,0)*0.2 + (vols[-1]/vols[:-1].mean()*100)*0.1 + max(mom,0)*0.2
            bucket = int(score//10)*10; prob = prob_map.get(bucket,0)
            candidates.append((code, price, score, prob))
        except:
            continue
    return candidates

# ─── 7) 추천 로직 ───────────────────────────────────────────────────
def recommend(prob_map, top_n=5):
    # 5분봉 기반 개별 알림
    cands = collect_5min_candidates(True, prob_map)
    if not cands:
        cands = collect_5min_candidates(False, prob_map)
    best5 = sorted(cands, key=lambda x:x[2], reverse=True)[:top_n]
    now = datetime.now()

    # 개별 알림 조건
    for code, price, score, prob in best5:
        if (score >= 80 and prob*100 >= 70) or score >= 90 or prob*100 >= 90:
            tp = price * (1 + TAKE_PROFIT_PCT/100)
            sl = price * (1 - STOP_LOSS_PCT/100)
            send_telegram(f"🔔 {code} | 가격:{price:,.2f} | 점수:{score:.1f} | 확률:{prob*100:.1f}% | 익절:{tp:,.2f} | 손절:{sl:,.2f}")

    # 하루 1회: 오전 9시 5분 일봉 기반 Top5 알림
    if now.hour == 9 and now.minute == 5:
        daily = []
        for code in CANDIDATES:
            try:
                df = pyupbit.get_ohlcv(code, "day", count=30)
                if df is None or len(df) < 30:
                    continue
                closes = df['close'].values; vols = df['volume'].values
                rsi   = compute_rsi(closes)
                macd  = compute_macd_hist(closes)
                mom   = compute_momentum(closes[-15:])
                score = (100 - rsi)*0.5 + max(macd,0)*0.2 + (vols[-1]/vols[:-1].mean()*100)*0.1 + max(mom,0)*0.2
                daily.append((code, score, rsi, macd, mom))
            except:
                continue
        top_daily = sorted(daily, key=lambda x:x[1], reverse=True)[:top_n]
        msg = f"📈 오늘 일봉 Top{top_n} 추천 (오전9시5분)\n"
        for i,(c,sc,r,ma,mo) in enumerate(top_daily,1):
            msg += f"{i}. {c} | 점수:{sc:.1f} | RSI:{r:.1f} | MACD:{ma:.2f} | 모멘텀:{mo:.2f}%\n"
        send_telegram(msg)

# ─── 8) 메인 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    seed_initial_prices()
    print("백테스트 중…(약 1~2분 소요)")
    prob_map = backtest_probabilities(CANDIDATES, samples=100)
    import threading
    t = threading.Thread(target=start_ws, daemon=True)
    t.start()
    time.sleep(30)
    while True:
        recommend(prob_map)
        time.sleep(300)
