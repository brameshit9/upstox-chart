# =========================================================
# SMART MONEY INTRADAY SCANNER — Upstox + Streamlit  (v11)
# =========================================================
# Rewritten to actually run on Streamlit, using the real Upstox
# REST API (OAuth2 login flow + v3 intraday candle API) instead
# of the undefined `fetch_intraday()` / matplotlib+IPython loop
# from the previous version.
#
# SETUP
# -----
# 1. Create an app at https://account.upstox.com/developer/apps
#    - Redirect URI must exactly match what you put in secrets
#      below (e.g. the URL of your deployed Streamlit app).
# 2. Put these in Streamlit secrets (.streamlit/secrets.toml
#    locally, or "Settings -> Secrets" on Streamlit Cloud):
#
#       UPSTOX_API_KEY = "your_api_key"
#       UPSTOX_API_SECRET = "your_api_secret"
#       UPSTOX_REDIRECT_URI = "https://your-app.streamlit.app"
#
# 3. `pip install -r requirements.txt`
# 4. `streamlit run app.py`
# =========================================================

import gzip
import io
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytz

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ============================== CONFIG ==============================
IST = pytz.timezone("Asia/Kolkata")

UPSTOX_BASE = "https://api.upstox.com"
LOGIN_DIALOG_URL = f"{UPSTOX_BASE}/v2/login/authorization/dialog"
TOKEN_URL = f"{UPSTOX_BASE}/v2/login/authorization/token"
INSTRUMENT_MASTER_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
)

SWING_LB = 5
ATR_PERIOD = 14
ATR_BIG_MULT = 1.8
OBV_SLOPE_N = 5
SMC_IMPULSE = 1.2
INTRADAY_INTERVAL_MIN = "1"  # 1-minute candles via v3 intraday API

# Colour palette
C_BULL, C_BEAR, C_NEUTRAL = "#1D9E75", "#D85A30", "#888780"
C_BULL_BG, C_BEAR_BG, C_NEUT_BG = "#E8F7F1", "#FAECE7", "#F4F3EF"
C_MUTED, C_BOS, C_BIGC, C_CHOCH = "#9A9590", "#2176AE", "#D4A000", "#7B5EA7"

# NSE trading symbols to scan (subset shown; add/remove freely in the UI)
STOCKS = [
    "SHRIRAMFIN", "BHARTIARTL", "AXISBANK", "SUNPHARMA", "CIPLA", 
    "HDFCLIFE", "APOLLOHOSP", "JIOFIN", "LT", "TMPV", 
    "ITC", "ICICIBANK", "INDIGO", "BAJAJ-AUTO", "NESTLEIND", 
    "BAJAJFINSV", "TATASTEEL", "ADANIPORTS", "DRREDDY", "GRASIM", 
    "ONGC", "TRENT", "HDFCBANK", "ADANIENT", "KOTAKBANK", 
    "JSWSTEEL", "ASIANPAINT", "SBILIFE", "MARUTI", "RELIANCE", 
    "EICHERMOT", "ULTRACEMCO", "HINDUNILVR", "SBIN", "MAXHEALTH", 
    "BAJFINANCE", "TITAN", "COALINDIA", "POWERGRID", "NTPC", 
    "TATACONSUM", "M&M", "HINDALCO", "BEL", "ETERNAL", 
    "TCS", "HCLTECH", "WIPRO", "INFY", "TECHM"
]

st.set_page_config(page_title="Smart Money Intraday Scanner", layout="wide")

# =========================================================
# AUTH  (Upstox OAuth2 authorization-code flow)
# =========================================================

def get_secret(name):
    return st.secrets.get(name) or ""


def build_login_url():
    api_key = get_secret("UPSTOX_API_KEY")
    redirect_uri = get_secret("UPSTOX_REDIRECT_URI")
    params = {
        "response_type": "code",
        "client_id": api_key,
        "redirect_uri": redirect_uri,
        "state": "smc_scanner",
    }
    query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"{LOGIN_DIALOG_URL}?{query}"


def exchange_code_for_token(code: str):
    data = {
        "code": code,
        "client_id": get_secret("UPSTOX_API_KEY"),
        "client_secret": get_secret("UPSTOX_API_SECRET"),
        "redirect_uri": get_secret("UPSTOX_REDIRECT_URI"),
        "grant_type": "authorization_code",
    }
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(TOKEN_URL, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()  # contains access_token


def ensure_authenticated():
    """Populates st.session_state['access_token']. Returns True if logged in."""
    if "access_token" in st.session_state and st.session_state["access_token"]:
        return True

    missing = [
        k for k in ("UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI")
        if not get_secret(k)
    ]
    if missing:
        st.error(
            "Missing Streamlit secrets: " + ", ".join(missing) +
            ". Add them under Settings -> Secrets (see README)."
        )
        st.stop()

    params = st.query_params
    code = params.get("code")
    if code:
        try:
            token_json = exchange_code_for_token(code)
            st.session_state["access_token"] = token_json["access_token"]
            st.session_state["user_name"] = token_json.get("user_name", "")
            st.query_params.clear()
            return True
        except Exception as e:
            st.error(f"Login failed while exchanging code for token: {e}")
            st.query_params.clear()
            return False

    login_url = build_login_url()
    st.info("Log in with your Upstox account to fetch live market data.")
    st.link_button("🔐 Log in with Upstox", login_url, type="primary")
    return False


def upstox_headers():
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {st.session_state['access_token']}",
    }

# =========================================================
# INSTRUMENT MASTER  (symbol -> instrument_key)
# =========================================================

@st.cache_data(ttl=24 * 3600, show_spinner="Downloading NSE instrument master…")
def load_instrument_master():
    r = requests.get(INSTRUMENT_MASTER_URL, timeout=60)
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    records = json.loads(raw)
    mapping = {}
    for rec in records:
        if rec.get("segment") == "NSE_EQ" and rec.get("instrument_type") == "EQ":
            mapping[rec["trading_symbol"]] = rec["instrument_key"]
    return mapping

# =========================================================
# DATA FETCH — Upstox v3 intraday candle API
# =========================================================

def fetch_intraday(instrument_key: str, interval_min: str = INTRADAY_INTERVAL_MIN):
    url = f"{UPSTOX_BASE}/v3/historical-candle/intraday/{instrument_key}/minutes/{interval_min}"
    resp = requests.get(url, headers=upstox_headers(), timeout=15)
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}: {resp.text[:200]}"

    payload = resp.json()
    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        return None, "no_candles"

    # Upstox returns candles newest-first: [ts, open, high, low, close, volume, oi]
    df = pd.DataFrame(
        candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "OI"]
    )
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, None

# =========================================================
# INDICATORS
# =========================================================

def compute_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df

def compute_ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def compute_rsi(s, period=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.rolling(period).mean()
    al = l.rolling(period).mean()
    return 100 - 100 / (1 + ag / al)

def compute_macd(s, fast=12, slow=26, sig=9):
    ml = compute_ema(s, fast) - compute_ema(s, slow)
    sl = compute_ema(ml, sig)
    return ml, sl

def compute_adx(df, period=14):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    pdm = hi.diff().clip(lower=0)
    ndm = (-lo.diff()).clip(lower=0)
    pdm = pdm.where(pdm > ndm, 0.0)
    ndm = ndm.where(ndm > pdm, 0.0)
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100 * pdm.rolling(period).mean() / atr
    ndi = 100 * ndm.rolling(period).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.rolling(period).mean(), atr

def compute_obv(df):
    direction = np.sign(df["Close"].diff().fillna(0))
    vol = df["Volume"].fillna(0)
    return (direction * vol).cumsum()

def compute_cvd_proxy(df):
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    buy_vol = (((df["Close"] - df["Low"]) / hl) * df["Volume"]).fillna(0)
    sell_vol = (((df["High"] - df["Close"]) / hl) * df["Volume"]).fillna(0)
    return (buy_vol - sell_vol).cumsum()

def compute_atr(df, period=ATR_PERIOD):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =========================================================
# SMART MONEY CONCEPTS
# =========================================================

def find_swings(df, lb=SWING_LB):
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    sh_idx, sl_idx = [], []
    for i in range(lb, n - lb):
        wh = highs[i - lb: i + lb + 1]
        wl = lows[i - lb: i + lb + 1]
        if highs[i] == wh.max():
            sh_idx.append(i)
        if lows[i] == wl.min():
            sl_idx.append(i)
    return sh_idx, sl_idx

def detect_choch_bos(df, lb=SWING_LB):
    result = dict(choch=None, bos=None, last_sh=None, last_sl=None, trend="ranging")
    sh_idx, sl_idx = find_swings(df, lb)
    if len(sh_idx) < 2 or len(sl_idx) < 2:
        return result

    sh_prices = df["High"].iloc[sh_idx].values
    sl_prices = df["Low"].iloc[sl_idx].values

    hh, lh = sh_prices[-1] > sh_prices[-2], sh_prices[-1] < sh_prices[-2]
    hl, ll = sl_prices[-1] > sl_prices[-2], sl_prices[-1] < sl_prices[-2]

    trend = "up" if (hh and hl) else ("down" if (lh and ll) else "ranging")
    result["trend"] = trend

    last_sh, last_sl = sh_prices[-1], sl_prices[-1]
    result["last_sh"], result["last_sl"] = round(last_sh, 2), round(last_sl, 2)

    close = df["Close"].iloc[-1]
    if trend == "up" and close > last_sh:
        result["bos"] = "bull"
    elif trend == "down" and close < last_sl:
        result["bos"] = "bear"

    if trend == "down" and close > last_sh:
        result["choch"] = "bull"
    elif trend == "up" and close < last_sl:
        result["choch"] = "bear"

    return result

def detect_order_block(df, atr_series, lb=SWING_LB, impulse_mult=SMC_IMPULSE):
    result = dict(ob_type=None, ob_high=None, ob_low=None, ob_fresh=False)
    if len(df) < lb + 5:
        return result

    closes, opens = df["Close"].values, df["Open"].values
    highs, lows = df["High"].values, df["Low"].values
    atrs = atr_series.values
    n = len(df)

    for i in range(n - 2, lb, -1):
        atr_val = atrs[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        move = abs(closes[i] - opens[i])

        if closes[i] > opens[i] and move > impulse_mult * atr_val:
            for j in range(i - 1, max(i - 10, 0), -1):
                if closes[j] < opens[j]:
                    fresh = closes[-1] > highs[j]
                    return dict(ob_type="bull", ob_high=round(highs[j], 2),
                                ob_low=round(lows[j], 2), ob_fresh=fresh)

        if closes[i] < opens[i] and move > impulse_mult * atr_val:
            for j in range(i - 1, max(i - 10, 0), -1):
                if closes[j] > opens[j]:
                    fresh = closes[-1] < lows[j]
                    return dict(ob_type="bear", ob_high=round(highs[j], 2),
                                ob_low=round(lows[j], 2), ob_fresh=fresh)

    return result

# =========================================================
# ANALYZE ONE STOCK
# =========================================================

def analyze_stock(name, instrument_key, err_rows):
    try:
        df, err_note = fetch_intraday(instrument_key)
        if df is None:
            err_rows.append((name, instrument_key, err_note or "unknown"))
            return None
        if len(df) < ATR_PERIOD + SWING_LB + 5:
            err_rows.append((name, instrument_key, "not_enough_bars_yet"))
            return None

        has_vol = df["Volume"].sum() > 0
        df = compute_vwap(df) if has_vol else df.assign(VWAP=df["Close"].expanding().mean())

        df["EMA9"] = compute_ema(df["Close"], 9)
        df["RSI"] = compute_rsi(df["Close"], 14)
        df["MACD"], df["MACD_Sig"] = compute_macd(df["Close"])
        df["ADX"], df["ATR_raw"] = compute_adx(df)
        df["ATR"] = compute_atr(df)
        df["OBV"] = compute_obv(df) if has_vol else pd.Series(0, index=df.index)
        df["CVD"] = compute_cvd_proxy(df) if has_vol else pd.Series(0, index=df.index)

        last = df.iloc[-1]
        price, vwap, ema9 = last["Close"], last["VWAP"], last["EMA9"]
        rsi, macd, macd_s = last["RSI"], last["MACD"], last["MACD_Sig"]
        adx, atr = last["ADX"], last["ATR"]

        obv_now = df["OBV"].iloc[-1]
        obv_prev = df["OBV"].iloc[-OBV_SLOPE_N] if len(df) > OBV_SLOPE_N else df["OBV"].iloc[0]
        obv_rising = bool(obv_now > obv_prev)

        cvd_raw = df["CVD"].iloc[-1]
        cvd_val = 0 if pd.isna(cvd_raw) else round(cvd_raw, 0)
        cvd_bull = cvd_val > 0

        body = abs(last["Close"] - last["Open"])
        big_candle = bool(not pd.isna(atr) and atr > 0 and body > ATR_BIG_MULT * atr)
        bc_dir = 1 if last["Close"] >= last["Open"] else -1

        smc = detect_choch_bos(df)
        ob = detect_order_block(df, df["ATR"])

        vol_raw = last["Volume"] if has_vol else 0
        volume = 0 if pd.isna(vol_raw) else int(vol_raw)

        checks_bull = {
            "Price > VWAP": price > vwap, "Price > EMA9": price > ema9,
            "RSI > 50": rsi > 50, "MACD > Signal": macd > macd_s, "ADX > 20": adx > 20,
        }
        checks_bear = {
            "Price < VWAP": price < vwap, "Price < EMA9": price < ema9,
            "RSI < 50": rsi < 50, "MACD < Signal": macd < macd_s, "ADX > 20": adx > 20,
        }
        bull, bear = all(checks_bull.values()), all(checks_bear.values())
        signal = "BULLISH" if bull else ("BEARISH" if bear else "NEUTRAL")
        checks = checks_bull if signal in ("BULLISH", "NEUTRAL") else checks_bear

        return dict(
            name=name, instrument_key=instrument_key, signal=signal,
            price=round(price, 2), vwap=round(vwap, 2), ema9=round(ema9, 2),
            rsi=round(rsi, 1), macd=round(macd, 4), macd_sig=round(macd_s, 4),
            adx=round(adx, 1), atr=round(atr, 2),
            obv_rising=obv_rising, cvd_val=int(cvd_val), cvd_bull=cvd_bull, volume=volume,
            big_candle=big_candle, bc_dir=bc_dir,
            choch=smc["choch"], bos=smc["bos"], trend=smc["trend"],
            last_sh=smc["last_sh"], last_sl=smc["last_sl"],
            ob_type=ob["ob_type"], ob_high=ob["ob_high"], ob_low=ob["ob_low"], ob_fresh=ob["ob_fresh"],
            checks=checks, df=df,
        )
    except Exception as e:
        err_rows.append((name, instrument_key, f"exception:{e}"))
        return None

# =========================================================
# PLOTLY CHART — 4-panel, one per stock
# =========================================================

def build_chart(r):
    df = r["df"]
    sig = r["signal"]
    accent = C_BULL if sig == "BULLISH" else (C_BEAR if sig == "BEARISH" else C_NEUTRAL)
    bg = C_BULL_BG if sig == "BULLISH" else (C_BEAR_BG if sig == "BEARISH" else C_NEUT_BG)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.17, 0.17, 0.16], vertical_spacing=0.03,
        subplot_titles=("", "RSI", "MACD", "OBV / CVD"),
    )

    fig.add_trace(go.Candlestick(
        x=df["Datetime"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=C_BULL, decreasing_line_color=C_BEAR, name="Price",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["VWAP"], line=dict(color=C_BOS, width=1.6), name="VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA9"], line=dict(color=C_BIGC, width=1.2), name="EMA9"), row=1, col=1)

    rsi_col = C_BULL if r["rsi"] > 50 else C_BEAR
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], line=dict(color=rsi_col, width=1.2), name="RSI", showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line=dict(color=C_BEAR, dash="dot", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=C_BULL, dash="dot", width=1), row=2, col=1)

    macd_hist = df["MACD"] - df["MACD_Sig"]
    hist_colors = [C_BULL if v >= 0 else C_BEAR for v in macd_hist]
    fig.add_trace(go.Bar(x=df["Datetime"], y=macd_hist, marker_color=hist_colors, opacity=0.5, name="Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], line=dict(color=C_BOS, width=1.1), name="MACD", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD_Sig"], line=dict(color=C_BIGC, width=1.1), name="Signal", showlegend=False), row=3, col=1)

    def norm(s):
        s = s.values.astype(float)
        peak = np.nanmax(np.abs(s))
        return s / peak if np.isfinite(peak) and peak != 0 else np.zeros_like(s)

    obv_col = C_BULL if r["obv_rising"] else C_BEAR
    cvd_col = C_BULL if r["cvd_bull"] else C_BEAR
    fig.add_trace(go.Scatter(x=df["Datetime"], y=norm(df["OBV"]), line=dict(color=obv_col, width=1.2), name="OBV", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=norm(df["CVD"]), line=dict(color=cvd_col, width=1.1, dash="dash"), name="CVD", showlegend=False), row=4, col=1)

    pn = sum(r["checks"].values())
    fig.update_layout(
        title=dict(text=f"{r['name']}   ₹{r['price']:.2f}   {sig} {pn}/5", font=dict(color=accent, size=16)),
        height=650, margin=dict(l=40, r=20, t=50, b=20),
        plot_bgcolor=bg, paper_bgcolor="#FFFFFF",
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    return fig

# =========================================================
# STREAMLIT UI
# =========================================================

st.title("📡 Smart Money Intraday Scanner")
st.caption("Live NSE intraday data via Upstox • VWAP / EMA9 / RSI / MACD / ADX • OBV & CVD flow • CHoCH / BOS / Order Blocks")

if not ensure_authenticated():
    st.stop()

with st.sidebar:
    st.success(f"Logged in{' as ' + st.session_state.get('user_name') if st.session_state.get('user_name') else ''}")
    if st.button("Log out"):
        st.session_state.pop("access_token", None)
        st.rerun()

    st.divider()
    selected = st.multiselect("Stocks to scan", STOCKS, default=STOCKS[:12])
    refresh_secs = st.slider("Auto-refresh (seconds)", 15, 300, 60, step=15)
    auto_on = st.checkbox("Enable auto-refresh", value=True)
    run_now = st.button("🔄 Scan now", type="primary")

if auto_on and HAS_AUTOREFRESH:
    st_autorefresh(interval=refresh_secs * 1000, key="scanner_autorefresh")
elif auto_on and not HAS_AUTOREFRESH:
    st.sidebar.warning("Install `streamlit-autorefresh` for auto-refresh (see requirements.txt).")

if not selected:
    st.info("Pick at least one stock from the sidebar.")
    st.stop()

instrument_map = load_instrument_master()

err_rows = []
results = []
progress = st.progress(0.0, text="Fetching intraday data…")
for i, name in enumerate(selected):
    ikey = instrument_map.get(name)
    if not ikey:
        err_rows.append((name, "?", "symbol_not_found_in_instrument_master"))
    else:
        results.append(analyze_stock(name, ikey, err_rows))
        time.sleep(0.1)  # be gentle on rate limits
    progress.progress((i + 1) / len(selected))
progress.empty()

valid = [r for r in results if r]
ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
st.caption(f"Last scan: {ts}  •  {len(valid)}/{len(selected)} fetched OK")

if err_rows:
    with st.expander(f"⚠️ Fetch issues ({len(err_rows)})"):
        st.dataframe(pd.DataFrame(err_rows, columns=["name", "instrument_key", "reason"]), hide_index=True)

if not valid:
    st.warning("No data fetched yet. If the market is closed, intraday candles may be empty.")
    st.stop()

# ---- Summary table ----
summary_rows = []
for r in valid:
    summary_rows.append(dict(
        Stock=r["name"], Signal=r["signal"], Price=r["price"], RSI=r["rsi"], ADX=r["adx"],
        ATR=r["atr"], OBV=("↑" if r["obv_rising"] else "↓"),
        CVD=("+" if r["cvd_bull"] else "") + f"{r['cvd_val']:,}",
        Trend=r["trend"], CHoCH=(r["choch"] or "—"), BOS=(r["bos"] or "—"),
        Score=f"{sum(r['checks'].values())}/5",
        BigCandle=("⚡" if r["big_candle"] else ""),
    ))
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# ---- Alert callouts ----
col1, col2, col3 = st.columns(3)
with col1:
    big = [r for r in valid if r["big_candle"]]
    st.markdown(f"**⚡ Big candles ({len(big)})**")
    for r in big:
        st.write(f"{r['name']} — {'Buy' if r['bc_dir']==1 else 'Sell'} impulse @ ₹{r['price']:.2f}")
with col2:
    ch = [r for r in valid if r["choch"]]
    st.markdown(f"**🔀 CHoCH flips ({len(ch)})**")
    for r in ch:
        st.write(f"{r['name']} — {'Bullish ↑' if r['choch']=='bull' else 'Bearish ↓'}")
with col3:
    ob = [r for r in valid if r["ob_type"] and r["ob_fresh"]]
    st.markdown(f"**🟧 Fresh order blocks ({len(ob)})**")
    for r in ob:
        st.write(f"{r['name']} — {r['ob_type']} zone {r['ob_low']}–{r['ob_high']}")

st.divider()

# ---- Charts, one per stock ----
sort_choice = st.radio("Chart order", ["Signal strength", "Alphabetical"], horizontal=True)
if sort_choice == "Signal strength":
    valid.sort(key=lambda r: sum(r["checks"].values()), reverse=True)
else:
    valid.sort(key=lambda r: r["name"])

for r in valid:
    st.plotly_chart(build_chart(r), use_container_width=True, key=f"chart_{r['name']}")
