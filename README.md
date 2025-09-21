# Shopee ROAS Dashboard

Streamlit app for visualising Shopee performance (ROAS, AOV, CPO, RPV, ORV) from a Google Sheet that updates every 10 minutes and creates a new column each hour (`D21 12:04`, `D21 13:07`, ...).

## Repo structure
```
.
├─ app.py                 # main app (already prepared)
├─ requirements.txt       # Python deps
├─ README.md              # this file
├─ .streamlit/
│  ├─ config.toml         # theme
│  └─ secrets.toml        # (local only) put ROAS_CSV_URL here when testing
└─ .gitignore
```

## Data format expected
- Column **`name`** = channel/store name.
- Many **time columns** like `D21 12:4`, `D21 13:07`, ... (the sheet overwrites the current hour every ~10 mins and adds a new column when the hour flips).
- Each time cell contains a comma string with **six values** in order:
  `budget,user,order,view,sale,ro`

The app will **unpivot** those time columns, parse the metrics, and compute:
`ROAS, AOV, CPO, RPV (sale/view), ORV (order/view)`.

## Prepare a Google Sheets CSV URL
1. Open your sheet tab and copy the **Sheet ID** and **gid**.
2. Build a **CSV export URL** like:
```
https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>
```
3. Share the sheet as **Anyone with the link – Viewer**.

## Run locally
```
pip install -r requirements.txt
# Option A: paste URL in the app sidebar
streamlit run app.py

# Option B: set an env var so the app loads data automatically
export ROAS_CSV_URL="https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>"
streamlit run app.py
```

> Local secret (optional): create `.streamlit/secrets.toml` with
> ```toml
> ROAS_CSV_URL="https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>"
> ```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (branch `main`).
2. Go to https://share.streamlit.io → **New app** → pick your repo.
3. Set **Main file path** to `app.py`.
4. (Recommended) In **Settings → Secrets**, add:
   ```
   ROAS_CSV_URL="https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>"
   ```
5. Deploy. The app will open with your data.

## Pages & Features
- **Overview**: KPI cards (Sales, Orders, Budget, ROAS, AOV, CPO, RPV, ORV), hourly trend, prime-hours heatmap, leaderboard, data table.
- **Channel**: multi‑axis line (Sales/Orders/Budget/ROAS), 24h bars, contribution %, time-series table.
- **Compare**: pick 2–4 channels; KPI table; %‑difference vs a baseline channel; small multiples.

## Tips
- If the app says “No time columns detected”, ensure your headers are like `D21 12:4` (letter + day + space + H:MM).
- Baseline for %Δ is **yesterday (same range)** by default.
- If you later want true 10‑minute granularity (not just hourly snapshots), add a small job to **append snapshots** into a DB (DuckDB/SQLite) whenever the sheet is fetched.

## Troubleshooting
- **403/404 on CSV URL**: check sharing permission and that `gid` is correct.
- **Weird numbers/Thai text in a cell**: the parser keeps the first 6 numeric values; non‑numeric is treated as missing.
- **Timezone**: timestamps are parsed as **Asia/Bangkok**.

---

Made for quick iteration — adjust the UI or metrics easily inside `app.py`.
