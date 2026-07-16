# Smart Money Intraday Scanner — Upstox + Streamlit

Live NSE intraday scanner: VWAP, EMA9, RSI, MACD, ADX, OBV, CVD proxy, and
Smart-Money-Concepts (CHoCH / BOS / Order Blocks), fetched directly from
Upstox and rendered in Streamlit.

## 1. Create an Upstox app

1. Go to <https://account.upstox.com/developer/apps> and create an app.
2. Set the **Redirect URI** to the exact URL your app will run at, e.g.:
   - Local: `http://localhost:8501`
   - Streamlit Cloud: `https://your-app-name.streamlit.app`
3. Note down the **API Key** (client_id) and **API Secret** (client_secret).

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Smart money intraday scanner"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`secrets.toml` is git-ignored on purpose — never commit real keys.

## 3. Configure secrets

**Local run:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real API key/secret/redirect URI
```

**Streamlit Community Cloud:**
1. Deploy the repo at <https://share.streamlit.io>.
2. App → Settings → Secrets → paste:
   ```toml
   UPSTOX_API_KEY = "your_api_key"
   UPSTOX_API_SECRET = "your_api_secret"
   UPSTOX_REDIRECT_URI = "https://your-app-name.streamlit.app"
   ```

## 4. Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **"Log in with Upstox"** in the app — you'll be redirected to Upstox's
login page, and after authenticating you land back on the app with a live
access token (valid until ~3:30 AM IST the same/next day, per Upstox's
token policy — you'll need to log in again after it expires).

## Notes

- Instrument symbols use NSE trading symbols (e.g. `RELIANCE`, `TCS`).
  Edit the `STOCKS` list in `app.py` to change the universe scanned.
- Data comes from Upstox's `v3/historical-candle/intraday` endpoint
  (1-minute candles for the current trading day). Outside NSE market
  hours (9:15 AM–3:30 PM IST, Mon–Fri) candles may be empty/stale.
- The instrument master (symbol → instrument_key mapping) is cached for
  24 hours to avoid re-downloading it every rerun.
- Streamlit Cloud apps sleep when idle and have an ephemeral filesystem,
  so this version doesn't write CSV logs to disk — everything is shown
  live in the UI. If you want persistent logging, add a database or
  cloud storage call inside `analyze_stock`/the main loop.
- Respect Upstox's API rate limits if you expand `STOCKS` significantly;
  the app fetches one stock at a time with a small delay between calls.
