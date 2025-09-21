
# Shopee ROAS Dashboard (Overview • Channel • Compare)
# ----------------------------------------------------
# How to run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Data options (sidebar):
#   1) Paste Google Sheets CSV export URL   e.g. https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv&gid=<GID>
#   2) Upload a CSV exported from your sheet
#
# Sheet shape expected (wide):
#   - Column 'name' = channel name
#   - One meta column like 'C_name,budget,user,order,view,sale,ro' (ignored)
#   - Many time columns like 'D21 12:4', 'D21 10:49' ... each cell = "budget,user,order,view,sale,ro"
#
# This app will:
#   - Unpivot time columns into long format
#   - Parse metrics into numeric columns
#   - Compute KPIs: ROAS, AOV, CPO, RPV, ORV
#   - Provide 3 pages: Overview, Channel, Compare
#
import os, re, io, math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date, timezone, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests

HAS_AGGRID = False  # ปิดไว้ก่อนให้รันชัวร์ ๆ (ค่อยเปิดทีหลัง)


st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:4

def is_time_col(col: str) -> bool:
    return TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, year=None, month=None, tz="Asia/Bangkok") -> pd.Timestamp:
    # hdr example: "D21 12:4" -> day=21, hour=12, minute=4
    # If month/year not given, use today's month/year (works for rolling daily dashboards)
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr.strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    if year is None: year = now.year
    if month is None: month = now.month
    try:
        ts = pd.Timestamp(year=year, month=month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        ts = pd.NaT
    return ts

def str_to_metrics_tuple(s: str):
    # s like "2314,17,47,1400,29,46,1024" but in sheet appears 6 values? From screenshots: budget,user,order,view,sale,ro
    # We'll keep first 6 numbers: budget,user,order,view,sale,ro
    # Also handle "0,0,0,0,0,0,2793" (extra trailing) or "ไม่มีแดน..." text -> None
    if not isinstance(s, str): 
        return (np.nan,)*6
    if not re.search(r"\d", s):
        return (np.nan,)*6
    # keep only digits and commas and dots
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]
    # If there are more than 6 numbers, take first 6
    nums = []
    for p in parts[:6]:
        try:
            nums.append(float(p))
        except:
            nums.append(np.nan)
    while len(nums) < 6:
        nums.append(np.nan)
    return tuple(nums[:6])

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # Identify columns
    cols = list(df_wide.columns)
    id_cols = []
    time_cols = []
    for c in cols:
        if c.strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
        else:
            # ignore meta columns
            pass
    if not time_cols:
        raise ValueError("No time columns detected. Please ensure headers like 'D21 12:4'.")
    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols, var_name="time_col", value_name="metrics")
    # Parse timestamp
    ts = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))
    df_melt["timestamp"] = ts  # ts ใส่โซนเวลาไว้แล้ว
    # Parse metrics
    metrics = df_melt["metrics"].apply(str_to_metrics_tuple)
    metrics_df = pd.DataFrame(metrics.tolist(), columns=["budget","user","order","view","sale","ro"])
    out = pd.concat([df_melt[["timestamp"] + id_cols], metrics_df], axis=1)
    out.rename(columns={"name":"channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    # Clean negatives or zeros for div safety
    for c in ["budget","user","order","view","sale","ro"]:
        # keep zeros; they'll be handled in kpi calcs
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # Derived KPIs
    out["ROAS"] = out["sale"] / out["budget"].replace(0, np.nan)
    out["AOV"]  = out["sale"] / out["order"].replace(0, np.nan)
    out["CPO"]  = out["budget"] / out["order"].replace(0, np.nan)
    out["RPV"]  = out["sale"] / out["view"].replace(0, np.nan)
    out["ORV"]  = out["order"] / out["view"].replace(0, np.nan)
    return out

def _fetch_csv(url: str) -> pd.DataFrame:
    import requests, io
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def load_data_from_sidebar():
    st.sidebar.header("Data source")
    default_url = os.environ.get("ROAS_CSV_URL", "")
    csv_url = st.sidebar.text_input(
        "Google Sheets CSV export URL (or leave blank to upload file):",
        value=st.session_state.get("csv_url", default_url),
        key="csv_url_input",
    )

    colA, colB = st.sidebar.columns(2)
    load_btn   = colA.button("Load data", type="primary")
    reload_btn = colB.button("Reload")
    clear_btn  = st.sidebar.button("Clear data")

    uploaded = st.sidebar.file_uploader("Upload CSV (exported from your sheet)", type=["csv"], key="uploader")

    # 0) Clear session
    if clear_btn:
        for k in ("df_wide", "source_type", "auto_loaded"):
            st.session_state.pop(k, None)

    # 1) Upload wins
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["df_wide"] = df
        st.session_state["source_type"] = "upload"
        return df, "upload"

    # 2) Explicit load / reload (from URL)
    if (load_btn or reload_btn) and csv_url:
        df = _fetch_csv(csv_url)
        st.session_state["df_wide"] = df
        st.session_state["source_type"] = "url"
        st.session_state["csv_url"] = csv_url
        st.session_state["auto_loaded"] = True
        return df, "url"

    # 3) Use existing data in session (สำคัญที่สุด กันเด้งกลับ)
    if "df_wide" in st.session_state:
        return st.session_state["df_wide"], st.session_state.get("source_type", "session")

    # 4) Auto-load ครั้งแรกถ้ามี URL ค่าเริ่มต้น
    if (csv_url or default_url) and "auto_loaded" not in st.session_state:
        url = csv_url or default_url
        df = _fetch_csv(url)
        st.session_state["df_wide"] = df
        st.session_state["source_type"] = "url"
        st.session_state["csv_url"] = url
        st.session_state["auto_loaded"] = True
        return df, "url"

    st.info("Paste the CSV URL then click **Load data**, or upload a CSV file.")
    return None, None


@st.cache_data(ttl=60*5, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def format_big(x):
    if pd.isna(x): return "-"
    try:
        return f"{x:,.0f}"
    except:
        return str(x)

def kpi_card(label, value, delta=None, help_text=None, col=None):
    target = st if col is None else col
    if delta is None:
        target.metric(label, format_big(value))
    else:
        target.metric(label, format_big(value), delta=f"{delta:+.1f}%")
    if help_text and col is None:
        st.caption(help_text)

# -----------------------------
# Main App
# -----------------------------

st.title("Shopee ROAS Dashboard")

df_wide, source_type = load_data_from_sidebar()
if df_wide is None:
    st.stop()

try:
    df_long = build_long(df_wide)
except Exception as e:
    st.error(f"Failed to parse sheet: {e}")
    st.write(df_wide.head())
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
channels = sorted(df_long["channel"].dropna().unique().tolist())
multi = st.sidebar.multiselect("Channels", options=channels, default=channels)
tz = "Asia/Bangkok"
min_ts = df_long["timestamp"].min()
max_ts = df_long["timestamp"].max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.sidebar.warning("No valid timestamps found.")
    st.stop()

date_min = min_ts.date()
date_max = max_ts.date()
d1, d2 = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
if isinstance(d1, (list, tuple)):
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

page = st.sidebar.radio("Page", ["Overview","Channel","Compare"])

# Apply filters
mask = (df_long["channel"].isin(multi)) & (df_long["timestamp"]>=start_ts) & (df_long["timestamp"]<=end_ts)
d = df_long.loc[mask].copy()
if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Utility aggregations
def aggregate_hourly(df):
    # Keep the latest snapshot within each hour for each channel
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("H")
    # latest snapshot per (channel, hour) by timestamp
    idx = df.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    hourly = df.loc[idx]
    return hourly

def compute_totals(df):
    totals = {}
    totals["Sales"] = df["sale"].sum()
    totals["Orders"] = df["order"].sum()
    totals["Budget"] = df["budget"].sum()
    totals["ROAS"] = (df["sale"].sum() / df["budget"].sum()) if df["budget"].sum()!=0 else np.nan
    totals["AOV"] = (df["sale"].sum() / df["order"].sum()) if df["order"].sum()!=0 else np.nan
    totals["CPO"] = (df["budget"].sum() / df["order"].sum()) if df["order"].sum()!=0 else np.nan
    totals["RPV"] = (df["sale"].sum() / df["view"].sum()) if df["view"].sum()!=0 else np.nan
    totals["ORV"] = (df["order"].sum() / df["view"].sum()) if df["view"].sum()!=0 else np.nan
    return totals

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev):
        return None
    if curr is None or pd.isna(curr): return None
    return (curr - prev) * 100.0 / prev

# Build hourly snapshot
hourly = aggregate_hourly(d)

# Baseline: yesterday same range
y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts - pd.Timedelta(days=1)
base_mask = (df_long["channel"].isin(multi)) & (df_long["timestamp"]>=y_start) & (df_long["timestamp"]<=y_end)
baseline_hourly = aggregate_hourly(df_long.loc[base_mask]) if not df_long.loc[base_mask].empty else None

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    totals = compute_totals(hourly)
    base_totals = compute_totals(baseline_hourly) if baseline_hourly is not None and not baseline_hourly.empty else {}
    cols = st.columns(8)
    labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
    for i,l in enumerate(labels):
        curr = totals.get(l, np.nan)
        prev = base_totals.get(l, np.nan) if base_totals else np.nan
        delta = pct_delta(curr, prev) if not pd.isna(prev) else None
        kpi_card(l, curr, delta=delta, col=cols[i])

    # Trend line
    st.markdown("### Trend by hour")
    # aggregate across selected channels
    trend = hourly.groupby("hour").agg({"sale":"sum","order":"sum","budget":"sum"}).reset_index()
    trend["ROAS"] = trend["sale"] / trend["budget"].replace(0, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["sale"], mode="lines+markers", name="Sales"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["order"], mode="lines+markers", name="Orders", yaxis="y2"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["ROAS"], mode="lines+markers", name="ROAS", yaxis="y3"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="ROAS", anchor="free", overlaying="y", side="right", position=1.0),
        legend=dict(orientation="h", y=-0.2),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.markdown("### Prime hours heatmap")
    metric = st.selectbox("Heatmap metric", ["Sales","ROAS","ORV"], index=0, key="heat_metric")
    tmp = hourly.copy()
    tmp["day"] = tmp["hour"].dt.date
    tmp["hr"]  = tmp["hour"].dt.hour
    agg_map = {"Sales":"sale","ROAS":"ROAS","ORV":"ORV"}
    heat = tmp.groupby(["day","hr"]).agg(val=(agg_map[metric],"mean")).reset_index()
    heat_pivot = heat.pivot(index="day", columns="hr", values="val").sort_index(ascending=False)
    fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(x="Hour", y="Day", color=metric))
    st.plotly_chart(fig_h, use_container_width=True)

    # Leaderboard
    st.markdown("### Leaderboard by channel")
    ldb = hourly.groupby("channel").agg(
        Sales=("sale","sum"),
        Orders=("order","sum"),
        Budget=("budget","sum")
    ).reset_index()
    ldb["ROAS"] = ldb["Sales"] / ldb["Budget"].replace(0, np.nan)
    ldb["CPO"]  = ldb["Budget"] / ldb["Orders"].replace(0, np.nan)
    ldb = ldb.sort_values(by="ROAS", ascending=False)
    if HAS_AGGRID:
        gb = GridOptionsBuilder.from_dataframe(ldb.round(2))
        gb.configure_default_column(resizable=True, filter=True, sortable=True)
        AgGrid(ldb.round(2), gridOptions=gb.build(), height=300)
    else:
        st.dataframe(ldb.round(2), use_container_width=True)

    # Raw table (current filters)
    st.markdown("### Data (hourly latest snapshot per channel)")
    show_cols = ["hour","channel","budget","user","order","view","sale","ROAS","AOV","CPO","RPV","ORV"]
    st.dataframe(hourly[show_cols].sort_values(["hour","channel"]).round(3), use_container_width=True)

# -----------------------------
# CHANNEL
# -----------------------------
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=channels, index=0)
    ch_df = hourly[hourly["channel"]==ch].copy()
    st.subheader(f"Channel • {ch}")
    # KPI row
    cols = st.columns(8)
    labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
    totals = compute_totals(ch_df)
    base_df = baseline_hourly[baseline_hourly["channel"]==ch] if baseline_hourly is not None else None
    base_totals = compute_totals(base_df) if base_df is not None and not base_df.empty else {}
    for i,l in enumerate(labels):
        curr = totals.get(l, np.nan)
        prev = base_totals.get(l, np.nan) if base_totals else np.nan
        delta = pct_delta(curr, prev) if not pd.isna(prev) else None
        kpi_card(l, curr, delta=delta, col=cols[i])

    # Multi-axis line
    st.markdown("### Multi-axis line")
    tr = ch_df.sort_values("hour")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["sale"], mode="lines+markers", name="Sales"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["order"], mode="lines+markers", name="Orders", yaxis="y2"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["budget"], mode="lines+markers", name="Budget", yaxis="y3"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["ROAS"], mode="lines+markers", name="ROAS", yaxis="y4"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="Budget", overlaying="y", side="right", position=1.0),
        yaxis4=dict(title="ROAS", overlaying="y", side="right", position=0.95),
        legend=dict(orientation="h", y=-0.3),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    # 24h bars
    st.markdown("### 24h distribution")
    tr["h"] = tr["hour"].dt.hour
    dbar = tr.groupby("h").agg(Sales=("sale","sum"), ORV=("ORV","mean"), RPV=("RPV","mean")).reset_index()
    figb = px.bar(dbar, x="h", y="Sales")
    st.plotly_chart(figb, use_container_width=True)
    st.caption("Bars show Sales/hour; hover to see ORV/RPV (mean).")

    # Contribution
    st.markdown("### Contribution % to total")
    total_sales = hourly.groupby("channel")["sale"].sum().sum()
    this_sales = ch_df["sale"].sum()
    st.write(f"Sales contribution of **{ch}**: {this_sales/total_sales*100 if total_sales else 0:.1f}%")

    # Table
    st.markdown("### Time series table")
    st.dataframe(ch_df[["hour","budget","user","order","view","sale","ROAS","AOV","CPO","RPV","ORV"]].round(3), use_container_width=True)

# -----------------------------
# COMPARE
# -----------------------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=channels, default=channels[:2])
    if len(pick) > 4:
        pick = pick[:4]
        st.info("Please pick at least 2 channels.")
        st.stop()
    st.subheader(f"Compare: {', '.join(pick)}")
    sub = hourly[hourly["channel"].isin(pick)].copy()
    # Radar-like via normalized metrics table in heatmap style
    kpis = sub.groupby("channel").agg(ROAS=("ROAS","mean"), AOV=("AOV","mean"), CPO=("CPO","mean"), RPV=("RPV","mean"), ORV=("ORV","mean")).reset_index()
    st.markdown("### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)
    # Diff vs baseline
    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS","sale","order","budget"], index=0)
    pivot = sub.pivot_table(index="hour", columns="channel", values=met, aggfunc="sum").sort_index()
    rel = (pivot.div(pivot[base], axis=0)-1.0)*100.0
    fig = go.Figure()
    for c in rel.columns:
        if c == base: 
            continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], mode="lines+markers", name=f"{c} vs {base}"))
    fig.update_layout(yaxis_title="% difference", xaxis_title="Hour", height=420, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Small multiples (ROAS)")
    sm = sub.pivot_table(index="hour", columns="channel", values="ROAS", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], mode="lines", name=c))
    fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig2, use_container_width=True)
