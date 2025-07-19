import os, time, uuid, json, requests, math
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyupbit, websocket
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
    except Exception as e:
        print("텔레그램 전송 실패:", e)

# ─── 1) 기본 설정 ────────────────────────────────────────────────────
CANDIDATES     = pyupbit.get_tickers(fiat="KRW")[:30]
latest_prices  = {}
price_history  = defaultdict(lambda: deque(maxlen=12))

# ─── 2) 초기 REST 시드 ─────────────────────────────────────────────────
def seed_initial_prices():
    print("초기 가격 시드 중…")
    for code in CANDIDATES:
        price = None
        # 1) pyupbit.get_current_price() 시도
        try:
            resp = pyupbit.get_current_price(code)
            if isinstance(resp, (int, float)):
                price = float(resp)
            elif isinstance(resp, dict) and 'trade_price' in resp:
                price = float(resp['trade_price'])
            elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                price = float(resp[0].get('trade_price', 0))
        except KeyError:
            pass
        except Exception as e:
            print(f"[PYUPBIT ERROR] {code}: {e}")

        # 2) 폴백: REST API 직접 호출
        if price is None:
            try:
                url = "https://api.upbit.com/v1/ticker"
                res = requests.get(url, params={"markets": code}, timeout=5).json()
                if isinstance(res, list) and res and 'trade_price' in res[0]:
                    price = float(res[0]['trade_price'])
            except Exception as e:
                print(f"[REST FALLBACK ERROR] {code}: {e}")

        # 검증 후 시드
        if price is None or price <= 0:
            print(f"[PRICE SKIP] {code}: 유효 가격 미획득")
            continue
        latest_prices[code] = price
        for _ in range(3):
            price_history[code].append((time.time(), price))

# ─── 3) 백테스트: 점수→확률 맵 생성 ────────────────────────────────────
def compute_rsi(closes, period=14):
    s     = pd.Series(closes, dtype=float)
    delta = s.diff().dropna()
    up    = delta.clip(lower=0).rolling(period).mean()
    down  = -delta.clip(upper=0).rolling(period).mean()
    rs    = up / down.replace(0, np.nan)
    rsi   = 100 - 100/(1+rs)
    return float(rsi.fillna(100).iloc[-1])

def backtest_probabilities(codes, samples=100, period=14):
    results = []
    for code in codes:
        df = pyupbit.get_ohlcv(code, "minute5", count=samples+period+1)
        if df is None or len(df) < samples+period+1:
            continue
        closes = df['close'].values
        for i in range(samples):
            window = closes[i:i+period+1]
            if len(window) < period+1:
                continue
            rsi = compute_rsi(window, period)
            score = 100 - rsi
            hit   = (closes[i+period] - closes[i+period-1]) > 0
            results.append((score, hit))
    if not results:
        return {b: 0.6 for b in range(0, 101, 10)}
    df = pd.DataFrame(results, columns=["score","hit"])
    df["bucket"] = (df["score"]//10)*10
    prob = df.groupby("bucket")["hit"].mean().to_dict()
    return {b: prob.get(b, 0.0) for b in range(0, 101, 10)}

# ─── 4) 지표 계산 함수 ─────────────────────────────────────────────────
def compute_macd_hist(closes):
    s = pd.Series(closes, dtype=float)
    macd   = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    val = (macd - signal).iloc[-1]
    return float(pd.Series([val]).fillna(0)[0])

def compute_momentum(hist_q):
    if len(hist_q) < 2:
        return 0.0
    p0, p1 = hist_q[0][1], hist_q[-1][1]
    return (p1 - p0)/p0 * 100 if p0 != 0 else 0.0

def compute_vwap(df):
    total_vol = df['volume'].sum()
    return float((df['close'] * df['volume']).sum() / total_vol) if total_vol > 0 else 0.0

def compute_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(pd.Series([val]).fillna(0)[0])

def compute_bb_width(df, period=20):
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + 2*std
    lower = sma - 2*std
    width = (upper - lower).iloc[-1]
    base = sma.iloc[-1]
    return float((width/base*100) if base > 0 else 0.0)

def compute_order_imbalance(code, depth=5):
    try:
        resp = pyupbit.get_orderbook(code)
        if isinstance(resp, list) and resp and 'orderbook_units' in resp[0]:
            units = resp[0]['orderbook_units']
        elif isinstance(resp, dict) and 'orderbook_units' in resp:
            units = resp['orderbook_units']
        else:
            return 0.0
        bids = sum(u.get('bid_size', 0) or u.get('bid_quantity', 0) for u in units[:depth])
        asks = sum(u.get('ask_size', 0) or u.get('ask_quantity', 0) for u in units[:depth])
        total = bids + asks
        return (bids - asks)/total * 100 if total > 0 else 0.0
    except Exception:
        return 0.0

# ─── 5) 후보 수집 & 점수화 ────────────────────────────────────────────
def collect_and_score(prob_map, strict=True):
    now    = datetime.now()
    scored = []
    for code, price in latest_prices.items():
        df = pyupbit.get_ohlcv(code, "minute5", count=max(20,15)+1)
        if df is None or len(df) < 20:
            continue
        if strict and (now - df.index[0].to_pydatetime() < timedelta(days=7)):
            continue
        if strict and df['volume'].iloc[-1] < df['volume'].mean():
            continue

        closes = df['close'].values
        # 지표 계산
        rsi  = compute_rsi(closes)
        macd = compute_macd_hist(closes)
        mom  = compute_momentum(price_history[code])
        vwap = compute_vwap(df.tail(20))
        atr  = compute_atr(df)
        bb_w = compute_bb_width(df)
        obi  = compute_order_imbalance(code)

        # NaN/무한 방지
        metrics = [rsi, macd, mom, vwap, atr, bb_w, obi]
        if any(math.isnan(x) or not math.isfinite(x) for x in metrics):
            continue

        score = (
            (100 - rsi)*0.30 +
            max(macd,0)*0.15 +
            (df['volume'].iloc[-1]/df['volume'].mean()*100)*0.10 +
            max(mom,0)*0.10 +
            abs(price - vwap)/vwap*100*0.10 +
            atr/price*100*0.10 +
            bb_w*0.05 +
            max(obi,0)*0.10
        )
        if math.isnan(score) or not math.isfinite(score):
            continue

        safe_score = score
        bucket     = int(safe_score//10)*10
        prob       = prob_map.get(bucket, 0.6)
        scored.append((code, price, score, prob))

    return scored

# ─── 6) 추천 & 알림 로직 ─────────────────────────────────────────────
def recommend(prob_map, top_n=3):
    cands = collect_and_score(prob_map, strict=True)
    if not cands:
        cands = collect_and_score(prob_map, strict=False)
    if not cands:
        cands = collect_and_score(prob_map, strict=None)

    best = sorted(cands, key=lambda x: x[2], reverse=True)[:top_n]
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(now, f"Top{top_n} 추천:")
    for i, (c, p, sc, pr) in enumerate(best, 1):
        tp = p*(1+TAKE_PROFIT_PCT/100)
        sl = p*(1-STOP_LOSS_PCT/100)
        line = (f"{i}. {c} | 가격:{p:,.0f} | 점수:{sc:.1f} | 확률:{pr*100:.1f}%"
                f" | 익절:{tp:,.0f} | 손절:{sl:,.0f}")
        print(line)
        if (sc>=80 and pr*100>=70) or sc>=90 or pr*100>=90:
            send_telegram("🔔 알림\n"+line)
    print("-"*50)

# ─── 7) WebSocket 실시간 업데이트 & 실행 진입점 ────────────────────────
def on_message(ws, msg):
    d = json.loads(msg)
    code, price = d.get("code"), d.get("trade_price")
    if code and isinstance(price, (int, float)):
        latest_prices[code] = price
        price_history[code].append((time.time(), price))

def on_open(ws):
    ws.send(json.dumps([{"ticket":str(uuid.uuid4())},
                        {"type":"ticker","codes":CANDIDATES}]))

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://api.upbit.com/websocket/v1",
        on_open=on_open, on_message=on_message
    )
    ws.run_forever()

if __name__ == "__main__":
    seed_initial_prices()
    print("백테스트 중…(약 1~2분 소요)")
    prob_map = backtest_probabilities(CANDIDATES, samples=100, period=14)
    print("확률 맵:", prob_map)

    import threading
    threading.Thread(target=start_ws, daemon=True).start()
    time.sleep(30)  # WebSocket 시드 대기

    while True:
        recommend(prob_map, top_n=3)
        time.sleep(300)
