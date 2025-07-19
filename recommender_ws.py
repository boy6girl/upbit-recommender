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

# ─── 0) 환경 변수 로드 ─────────────────────────────────────────────
load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID  = os.getenv("CHAT_ID")

def send_telegram(message: str):
    """텔레그램으로 메시지 전송"""
    url     = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("텔레그램 전송 실패:", e)

# ─── 1) 설정 ─────────────────────────────────────────────────────────
CANDIDATES    = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices = {}
price_history = defaultdict(lambda: deque(maxlen=12))

# ─── 2) 초기 REST 시드 ────────────────────────────────────────────────
def seed_initial_prices():
    print("초기 가격 시드 중…")
    for code in CANDIDATES:
        try:
            price = pyupbit.get_current_price(code)
            if price is None or not isinstance(price, (int, float)):
                continue
            latest_prices[code] = price
            for _ in range(3):
                price_history[code].append((time.time(), price))
        except:
            continue

# ─── 3) 백테스트: score→확률 맵 생성 ─────────────────────────────────
def compute_rsi(closes, period=14):
    s     = pd.Series(closes)
    delta = s.diff().dropna()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    rs    = up / down.replace(0, np.nan)
    return (100 - 100/(1 + rs)).iloc[-1]

def backtest_probabilities(codes, samples=100):
    results = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "minute5", count=samples+16)
        if df is None or len(df) < samples+16:
            continue
        closes = df['close'].values
        for i in range(samples):
            window = closes[i:i+15]
            try:
                rsi = compute_rsi(window)
            except:
                continue
            score = 100 - rsi
            hit   = (closes[i+15] - closes[i+14]) > 0
            results.append((score, hit))
    if not results:
        return {b:0.6 for b in range(0,101,10)}
    df = pd.DataFrame(results, columns=["score","hit"])
    df["bucket"] = (df["score"]//10)*10
    prob = df.groupby("bucket")["hit"].mean().to_dict()
    return {b:prob.get(b,0.0) for b in range(0,101,10)}

# ─── 4) WebSocket 수집 ─────────────────────────────────────────────
def on_message(ws, msg):
    data  = json.loads(msg)
    code  = data.get("code")
    price = data.get("trade_price")
    if code and price is not None:
        latest_prices[code] = price
        price_history[code].append((time.time(), price))

def on_open(ws):
    ws.send(json.dumps([{"ticket":str(uuid.uuid4())},
                        {"type":"ticker","codes":CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_message=on_message, on_open=on_open
    )
    ws.run_forever()

# ─── 5) 지표 계산 ────────────────────────────────────────────────────
def compute_macd_hist(closes):
    s      = pd.Series(closes)
    macd   = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal).iloc[-1]

def compute_momentum(hist_q):
    if len(hist_q) < 2:
        return 0.0
    p0 = hist_q[0][1]; p1 = hist_q[-1][1]
    return ((p1 - p0) / p0) * 100

# ─── 6) 후보 수집 헬퍼 ─────────────────────────────────────────────
def _collect_candidates(strict, prob_map):
    now        = datetime.now()
    candidates = []
    for code, price in latest_prices.items():
        try:
            df = pyupbit.get_ohlcv(code, "minute5", count=15)
            if df is None or df.empty:
                continue
            # 신규 상장 필터
            if strict is True and (now - df.index[0].to_pydatetime() < timedelta(days=7)):
                continue
            closes = df['close'].values; vols = df['volume'].values
            # 거래량 필터
            if strict is True and (vols[-1] < vols[-15:].mean()):
                continue
            # 지표 계산
            rsi   = compute_rsi(closes)
            macd  = compute_macd_hist(closes)
            mom   = compute_momentum(price_history[code])
            score = (100 - rsi)*0.5 + max(macd,0)*0.2 \
                  + (vols[-1]/vols[-15:].mean()*100)*0.1 + max(mom,0)*0.2
            bucket = int(score//10)*10
            prob   = prob_map.get(bucket,0.6)
            candidates.append((code, price, score, rsi, macd, vols[-1], mom, prob))
        except:
            continue
    return candidates

# ─── 7) 추천 로직 ──────────────────────────────────────────────────
def recommend(prob_map, top_n=5):
    print(f"[DEBUG] 수집 종목 수: {len(latest_prices)}")
    # 1차: strict=True
    cands = _collect_candidates(True, prob_map)
    # 2차: strict=False
    if not cands:
        print("[WARN] 거래량 필터 완화")
        cands = _collect_candidates(False, prob_map)
    # 3차: strict=None
    if not cands:
        print("[WARN] 모든 필터 해제")
        cands = _collect_candidates(None, prob_map)

    best = sorted(cands, key=lambda x:x[2], reverse=True)[:top_n]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now, "Top5 추천:")
    for i,(c,p,sc,_,_,_,_,pr) in enumerate(best,1):
        # 가격 소수점2자리, 천 단위 콤마
        print(f"{i}. {c} | 가격:{p:,.2f} | 확률:{pr*100:.1f}% | 점수:{sc:.1f}")
        # 텔레그램 알림: 점수 ≥ 85
        if sc >= 85:
            send_telegram(f"🔔 점수 85 이상 종목\n{i}. {c} | 가격:{p:,.2f} | 확률:{pr*100:.1f}% | 점수:{sc:.1f}")
    print("-"*70)

# ─── 8) 실행 진입점 ────────────────────────────────────────────────
if __name__ == "__main__":
    seed_initial_prices()
    print("백테스트 중…(약 1~2분 소요)")
    prob_map = backtest_probabilities(CANDIDATES, samples=100)
    print("확률 맵:", prob_map)
    import threading
    t = threading.Thread(target=start_ws, daemon=True)
    t.start()
    time.sleep(30)
    while True:
        recommend(prob_map, top_n=5)
        time.sleep(300)
