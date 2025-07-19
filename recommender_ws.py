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

# ─── 설정 ─────────────────────────────────────────────────────────────
load_dotenv()
TG_TOKEN        = os.getenv("TELEGRAM_TOKEN")
CHAT_ID         = os.getenv("CHAT_ID")
TAKE_PROFIT_PCT = 1.0    # 익절 목표 +1%
STOP_LOSS_PCT   = 0.5    # 손절 한계 -0.5%
SCORE_WINDOW    = 30     # 일봉 점수 계산용 기간

# ─── 헬퍼 함수 ────────────────────────────────────────────────────────
def send_telegram(msg: str):
    """텔레그램 메시지 전송"""
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    except:
        pass

def compute_rsi(arr, period=14):
    s   = pd.Series(arr, dtype=float)
    d   = s.diff().dropna()
    up  = d.clip(lower=0).rolling(period).mean()
    dn  = -d.clip(upper=0).rolling(period).mean()
    rs  = up / dn.replace(0, np.nan)
    return float((100 - 100/(1+rs)).fillna(100).iloc[-1])

# ─── 초기 시드 & WebSocket ───────────────────────────────────────────
CANDIDATES    = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices = {}
price_history = defaultdict(lambda: deque(maxlen=12))

def seed_initial_prices():
    for code in CANDIDATES:
        price = None
        try:
            r = pyupbit.get_current_price(code)
            if isinstance(r, (int, float)): price = r
            elif isinstance(r, dict):      price = r.get("trade_price")
            elif isinstance(r, list) and r: price = r[0].get("trade_price")
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

def on_message(ws, msg):
    d = json.loads(msg)
    c, p = d.get("code"), d.get("trade_price")
    if c and isinstance(p, (int, float)):
        latest_prices[c] = p
        price_history[c].append((time.time(), p))

def on_open(ws):
    ws.send(json.dumps([{"ticket": str(uuid.uuid4())},
                        {"type": "ticker", "codes": CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_open=on_open, on_message=on_message
    )
    ws.run_forever()

# ─── 5분봉 전략 ───────────────────────────────────────────────────────
def backtest_probabilities(codes, samples=100, period=14):
    results = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "minute5", count=samples+period+1)
        if df is None or len(df) < samples+period+1: continue
        cl = df['close'].values
        for i in range(samples):
            window = cl[i:i+period+1]
            rsi    = compute_rsi(window, period)
            score  = 100 - rsi
            hit    = (cl[i+period] - cl[i+period-1]) > 0
            results.append((score, hit))
    if not results:
        return {b:0.6 for b in range(0,101,10)}
    df = pd.DataFrame(results, columns=["score","hit"])
    df["bucket"] = (df["score"]//10)*10
    prob = df.groupby("bucket")["hit"].mean().to_dict()
    return {b:prob.get(b,0.0) for b in range(0,101,10)}

def compute_macd_hist(arr):
    s    = pd.Series(arr, dtype=float)
    macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig  = macd.ewm(span=9).mean()
    return float((macd - sig).iloc[-1] if not np.isnan(macd.iloc[-1]-sig.iloc[-1]) else 0)

def compute_vwap(df):
    tot = df['volume'].sum()
    return float((df['close']*df['volume']).sum()/tot) if tot>0 else 0.0

def collect_and_score_5m(prob_map, strict=True):
    now    = datetime.now()
    scored = []
    for code, price in latest_prices.items():
        df = pyupbit.get_ohlcv(code, "minute5", count=20)
        if df is None or len(df)<20: continue
        if strict and (now - df.index[0].to_pydatetime() < timedelta(days=7)): continue
        if strict and df['volume'].iloc[-1] < df['volume'].mean(): continue
        arr   = df['close'].values
        rsi   = compute_rsi(arr)
        macd  = compute_macd_hist(arr)
        vwap  = compute_vwap(df)
        score = (100-rsi)*0.5 + max(macd,0)*0.3 + max((df['close'].iloc[-1]-vwap)/vwap*100,0)*0.2
        if math.isnan(score): continue
        bucket = int(score//10)*10
        prob   = prob_map.get(bucket, 0.6)
        scored.append((code, price, score, prob))
    return scored

def recommend_5m(prob_map, top_n=3):
    cands = collect_and_score_5m(prob_map, strict=True)
    if not cands:
        cands = collect_and_score_5m(prob_map, strict=False)
    best = sorted(cands, key=lambda x:x[2], reverse=True)[:top_n]

    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 초 포함
    lines = [f"{now} 🔔 5m Top{top_n}:"]
    for i,(c,p,sc,pr) in enumerate(best,1):
        tp = p*(1+TAKE_PROFIT_PCT/100)
        sl = p*(1-STOP_LOSS_PCT/100)
        lines.append(f"{i}. {c} | 종가:{p:,.0f}원 | 점수:{sc:.1f} | 확률:{pr*100:.1f}% | 익절:{tp:,.0f}원 | 손절:{sl:,.0f}원")
        lines.append("")
    full = "\n".join(lines).strip()
    print(full)
    # 텔레그램 전송 조건
    for _,_,sc,pr in best:
        if (sc>=80 and pr>=0.7) or sc>=90 or pr>=0.9:
            send_telegram(full)
            break

# ─── 일봉 전략 ───────────────────────────────────────────────────────
def compute_daily_score(df):
    close     = df['close']
    rsi       = compute_rsi(close.values)
    ema20     = close.ewm(span=20).mean().iloc[-1]
    ema50     = close.ewm(span=50).mean().iloc[-1]
    adx       = ((df['high']-df['low'])/df['close']).rolling(14).mean().iloc[-1]*100
    bb_w      = ((close.rolling(20).mean()+2*close.rolling(20).std()) -
                 (close.rolling(20).mean()-2*close.rolling(20).std())).iloc[-1] \
                / close.rolling(20).mean().iloc[-1]*100
    today     = df.iloc[-1]
    today_pct = max((today['close']-today['open'])/today['open']*100, 0)
    vol_score = df['volume'].iloc[-1]/df['volume'].tail(20).mean()*100
    return ((100-rsi)*0.2 +
            max((ema20-ema50)/ema50*100,0)*0.15 +
            adx*0.1 + bb_w*0.1 +
            today_pct*0.15 + vol_score*0.15)

def backtest_daily_probabilities(codes, lookback=250):
    results=[]
    count   = lookback + SCORE_WINDOW + 1
    for code in codes:
        df = pyupbit.get_ohlcv(code, "day", count=count)
        if df is None or len(df)<count: continue
        for i in range(lookback):
            window = df.iloc[i:i+SCORE_WINDOW]
            today  = window.iloc[-1]
            nxt    = df.iloc[i+SCORE_WINDOW]
            sc     = compute_daily_score(window)
            bucket = int(sc//10)*10
            hit    = nxt['close']>today['close']
            results.append((bucket, hit))
    if not results:
        return {b:0.6 for b in range(0,101,10)}
    dfp = pd.DataFrame(results, columns=["bucket","hit"])
    pm  = dfp.groupby("bucket")["hit"].mean().to_dict()
    return {b:pm.get(b,0.0) for b in range(0,101,10)}

def daily_recommend(codes, prob_map_day, top_n=3):
    scored=[]
    for code in codes:
        df = pyupbit.get_ohlcv(code, "day", count=SCORE_WINDOW)
        if df is None or df.empty: continue
        p      = df['close'].iloc[-1]
        sc     = compute_daily_score(df)
        bucket = int(sc//10)*10
        pr     = prob_map_day.get(bucket,0.0)
        scored.append((code, p, sc, pr))
    best = sorted(scored, key=lambda x:x[2], reverse=True)[:top_n]

    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 초 포함
    lines = [f"{now} 📈 Daily Top{top_n}:"]
    for i,(c,p,sc,pr) in enumerate(best,1):
        tp = p*(1+TAKE_PROFIT_PCT/100)
        sl = p*(1-STOP_LOSS_PCT/100)
        lines.append(f"{i}. {c} | 종가:{p:,.0f}원 | 점수:{sc:.1f} | 확률:{pr*100:.1f}% | 익절:{tp:,.0f}원 | 손절:{sl:,.0f}원")
        lines.append("")
    full = "\n".join(lines).strip()
    print(full)
    for _,_,sc,pr in best:
        if (sc>=80 and pr>=0.7) or sc>=90 or pr>=0.9:
            send_telegram(full)
            break

# ─── 실행 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed_initial_prices()
    prob_map_5m  = backtest_probabilities(CANDIDATES)
    prob_map_day = backtest_daily_probabilities(CANDIDATES)
    threading.Thread(target=start_ws, daemon=True).start()
    time.sleep(30)  # WebSocket 시드 대기

    daily_sent = False
    while True:
        recommend_5m(prob_map_5m, top_n=3)
        now = datetime.now()
        if now.hour == 9 and now.minute == 5 and not daily_sent:
            daily_recommend(CANDIDATES, prob_map_day, top_n=3)
            daily_sent = True
        if now.hour == 9 and now.minute == 6:
            daily_sent = False
        time.sleep(60)  # 1분마다 실행
