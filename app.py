# Shopee ROAS Dashboard (New KPIs: Sales / Orders / Ads / Ads_RO / Sale_RO)
# --------------------------------------------------------------------------
# Data source: Google Sheets CSV (gviz/tq?tqx=out:csv&sheet=<sheet_name>)
#   - Set a Secret on Streamlit Cloud: ROAS_CSV_URL="https://docs.google.com/.../gviz/tq?tqx=out:csv&sheet=sale_roai"
#   - Or set environment variable ROAS_CSV_URL
#
# Behaviors:
#   - Read only (no switching sheet). Left panel shows only Reload + Last refresh.
#   - Auto-refresh every 10 minutes.
#   - Parse time headers like "D21 12:4" (day=21 hour=12 minute=4) → timestamp (Asia/Bangkok)
#   - From the selected time columns (C/D/...): take first four numbers per cell:
#       sales, orders, ads, ads_ro
#     (Sale_RO is computed: total_sales / total_ads)
#   - %Δ under KPIs compares against the same time range "yesterday"
#   - Default date range = latest 3 calendar days (including today)
#   - Filters: Channels = contains "All" option (if All chosen → all channels)
#   - Channel page: pick one channel from ALL channels (ignores filter)
#
# --------------------------------------------------------------------------

import os, re, io, math, time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit.components.v1 import html

# -----------------------------------------------------------------------------
# Page config + Auto refresh (10 minutes)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Shopee ROAS", layout="wide")

def auto_refresh(interval_ms: int = 600000):
    """Auto reload the page using a tiny JS snippet."""
    html(f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {interval_ms});
        </script>
        """,
        height=0,
    )

auto_refresh(600000)  # 10 minutes

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{{1,2}}\s+\d{{1,2}}:\d{{1,2}}$")  # e.g. D21 12:4

def is_time_col(col: str) -> bool:
    return TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz="Asia/Bangkok") -> pd.Timestamp:
    """
    "D21 12:4" -> day=21, hour=12, minute=4 (assume current month/year)
    """
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr.strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        ts = pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        ts = pd.NaT
    return ts

def str_to_metrics_tuple(s: str):
    """
    Convert metrics cell into (sales, orders, ads, ads_ro).
    The sheet cell is a CSV string; we take the first 4 numeric tokens.
    (Sale_RO will be computed later as total_sales / total_ads)
    """
    if not isinstance(s, str):
        return (np.nan, np.nan, np.nan, np.nan)
    if not re.search(r"\d", s):
        return (np.nan, np.nan, np.nan, np.nan)

    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]

    vals = []
    for p in parts[:4]:  # sales, orders, ads, ads_ro
        try:
            vals.append(float(p))
        except:
            vals.append(np.nan)

    while len(vals) < 4:
        vals.append(np.nan)
    return tuple(vals[:4])

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    """
    Melt wide sheet into long format with columns:
      channel, timestamp, sales, orders, ads, ads_ro
    """
    cols = list(df_wide.columns)
    id_cols, time_cols = [], []
    for c in cols:
        if c.strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)

    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")

    ts = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))
    df_melt["timestamp"] = ts

    metrics = df_melt["metrics"].apply(str_to_metrics_tuple)
    metrics_df = pd.DataFrame(metrics.tolist(),
                              columns=["sales", "orders", "ads", "ads_ro"])

    out = pd.concat([df_melt[["timestamp"] + id_cols], metrics_df], axis=1)
    out.rename(columns={"name": "channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    for c in ["sales", "orders", "ads", "ads_ro"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # convenience per-row ratio (may be NaN)
    out["sale_ro_row"] = out["sales"] / out["ads"].replace(0, np.nan)

    return out

def aggregate_hourly(df):
    """Keep latest snapshot within each hour per channel."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("H")
    idx = df.sort_values("timestamp").groupby(["channel", "hour"]).tail(1).index
    return df.loc[idx]

def compute_totals(df):
    """
    New KPI definitions:
      Sales  = sum(sales)
      Orders = sum(orders)
      Ads    = sum(ads)
      Ads_RO = mean(ads_ro where ads_ro > 0)
      Sale_RO= (sum sales) / (sum ads)
    """
    t = {}
    if df is None or df.empty:
        for k in ["Sales", "Orders", "Ads", "Ads_RO", "Sale_RO"]:
            t[k] = np.nan
        return t

    t["Sales"]  = df["sales"].sum()
    t["Orders"] = df["orders"].sum()
    t["Ads"]    = df["ads"].sum()

    pos = df["ads_ro"] > 0
    t["Ads_RO"] = df.loc[pos, "ads_ro"].mean() if pos.any() else np.nan

    t["Sale_RO"] = (t["Sales"] / t["Ads"]) if (t["Ads"] and t["Ads"] != 0) else np.nan
    return t

def pct_delta(curr, prev):
    """% change vs baseline."""
    if prev in [0, None] or (isinstance(prev, float) and np.isnan(prev)):
        return None
    if curr is None or (isinstance(curr, float) and np.isnan(curr)):
        return None
    try:
        return (curr - prev) * 100.0 / prev
    except ZeroDivisionError:
        return None

# -----------------------------------------------------------------------------
# Data loading (from Secrets / env)
# -----------------------------------------------------------------------------

def get_csv_url() -> str:
    url = ""
    try:
        # Prefer Streamlit Secrets
        url = st.secrets["ROAS_CSV_URL"]
    except Exception:
        # Fallback to env
        url = os.environ.get("ROAS_CSV_URL", "")
    return url

@st.cache_data(ttl=300, show_spinner=True)  # cache 5 minutes
def load_wide_df(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

@st.cache_data(ttl=300, show_spinner=False)
def build_long_df(df_wide: pd.DataFrame) -> pd.DataFrame:
    return long_from_wide(df_wide)

def reload_data():
    st.cache_data.clear()
    st.session_state["last_reload"] = datetime.now()

# -----------------------------------------------------------------------------
# Sidebar (Reload only)
# -----------------------------------------------------------------------------
st.sidebar.header("Data")
if "last_reload" not in st.session_state:
    st.session_state["last_reload"] = datetime.now()

colR1, colR2 = st.sidebar.columns([1,1])
if colR1.button("Reload", use_container_width=True):
    reload_data()
    st.experimental_rerun()

colR2.write("")
colR2.caption(f"Last refresh: {st.session_state['last_reload'].strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------------------------------------------------------
# Load & Transform
# -----------------------------------------------------------------------------
csv_url = get_csv_url()
if not csv_url:
    st.error("Missing ROAS_CSV_URL. Please set it in Secrets or environment.")
    st.stop()

try:
    df_wide = load_wide_df(csv_url)
except Exception as e:
    st.error(f"Download failed: {e}")
    st.stop()

try:
    df_long = build_long_df(df_wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.write(df_wide.head())
    st.stop()

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
st.title("Shopee ROAS Dashboard")

# channels list (for filter) + Add 'All'
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
filter_options = ["All"] + all_channels
default_sel = ["All"]

st.sidebar.header("Filters")
selected_channels = st.sidebar.multiselect("Channels", options=filter_options, default=default_sel)

# Treat "All" as selecting everything
if ("All" in selected_channels) or (len(selected_channels) == 0):
    selected_channels_effective = all_channels
else:
    selected_channels_effective = selected_channels

# Default date range = latest 3 calendar days (including today)
min_ts = df_long["timestamp"].min()
max_ts = df_long["timestamp"].max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.warning("No valid timestamps found in the sheet.")
    st.stop()

date_max = max_ts.date()
date_min = (max_ts - pd.Timedelta(days=2)).date()  # 3 days window
d1, d2 = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=min_ts.date(),
    max_value=date_max
)
if isinstance(d1, (list, tuple)):
    d1, d2 = d1

tz = "Asia/Bangkok"
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

page = st.sidebar.radio("Page", ["Overview", "Channel"], index=0)

# Apply filters for the working dataset
mask = (df_long["channel"].isin(selected_channels_effective)) & \
       (df_long["timestamp"] >= start_ts) & (df_long["timestamp"] <= end_ts)
d = df_long.loc[mask].copy()

if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# Precompute hourly snapshot (latest per hour)
hourly = aggregate_hourly(d)

# Baseline = yesterday same range (still respecting the filter channels)
y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts   - pd.Timedelta(days=1)
base_mask = (df_long["channel"].isin(selected_channels_effective)) & \
            (df_long["timestamp"] >= y_start) & (df_long["timestamp"] <= y_end)
baseline_hourly = aggregate_hourly(df_long.loc[base_mask]) if not df_long.loc[base_mask].empty else None

# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview (All selected channels)")

    curr = compute_totals(hourly)
    prev = compute_totals(baseline_hourly) if baseline_hourly is not None else {}

    labels = ["Sales","Orders","Ads","Ads_RO","Sale_RO"]
    cols = st.columns(len(labels))
    for i, k in enumerate(labels):
        now = curr.get(k, np.nan)
        bef = prev.get(k, np.nan)
        delta = pct_delta(now, bef)
        # pretty format
        if k in ["Ads_RO", "Sale_RO"]:
            value_str = "-" if pd.isna(now) else f"{now:,.2f}"
        else:
            value_str = "-" if pd.isna(now) else f"{now:,.0f}"
        delta_str = None if delta is None else f"{delta:+.1f}%"
        cols[i].metric(k, value_str, delta_str)

    st.markdown("### Trend by hour (Sales / Orders / Ads)")
    tr = hourly.groupby("hour").agg(
        sales=("sales","sum"),
        orders=("orders","sum"),
        ads=("ads","sum"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["sales"],  mode="lines+markers", name="Sales"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["orders"], mode="lines+markers", name="Orders", yaxis="y2"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["ads"],    mode="lines+markers", name="Ads",    yaxis="y3"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="Ads",    overlaying="y", side="right", position=1.0),
        legend=dict(orientation="h", y=-0.2),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prime hours heatmap (Sales)")
    tmp = hourly.copy()
    tmp["day"] = tmp["hour"].dt.date
    tmp["hr"]  = tmp["hour"].dt.hour
    heat = tmp.groupby(["day","hr"]).agg(val=("sales","sum")).reset_index()
    heat_pivot = heat.pivot(index="day", columns="hr", values="val").sort_index(ascending=False)
    fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                      labels=dict(x="Hour", y="Day", color="Sales"))
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("### Data (hourly latest snapshot per channel)")
    show_cols = ["hour","channel","sales","orders","ads","ads_ro","sale_ro_row"]
    st.dataframe(hourly[show_cols].sort_values(["hour","channel"]).round(3),
                 use_container_width=True)

# -----------------------------------------------------------------------------
# CHANNEL
# -----------------------------------------------------------------------------
else:
    # IMPORTANT: list all channels from sheet (ignoring filter)
    ch = st.selectbox("Pick one channel (all channels)", options=all_channels, index=0)

    # current range for that channel
    ch_mask = (df_long["channel"] == ch) & \
              (df_long["timestamp"] >= start_ts) & (df_long["timestamp"] <= end_ts)
    ch_df = aggregate_hourly(df_long.loc[ch_mask])

    # baseline yesterday for that channel
    ch_base_mask = (df_long["channel"] == ch) & \
                   (df_long["timestamp"] >= y_start) & (df_long["timestamp"] <= y_end)
    base_ch = aggregate_hourly(df_long.loc[ch_base_mask]) if not df_long.loc[ch_base_mask].empty else None

    st.subheader(f"Channel • {ch}")

    curr = compute_totals(ch_df)
    prev = compute_totals(base_ch) if base_ch is not None else {}

    labels = ["Sales","Orders","Ads","Ads_RO","Sale_RO"]
    cols = st.columns(len(labels))
    for i, k in enumerate(labels):
        now = curr.get(k, np.nan)
        bef = prev.get(k, np.nan)
        delta = pct_delta(now, bef)
        if k in ["Ads_RO", "Sale_RO"]:
            value_str = "-" if pd.isna(now) else f"{now:,.2f}"
        else:
            value_str = "-" if pd.isna(now) else f"{now:,.0f}"
        delta_str = None if delta is None else f"{delta:+.1f}%"
        cols[i].metric(k, value_str, delta_str)

    st.markdown("### Multi-axis line (Sales / Orders / Ads)")
    tr = ch_df.sort_values("hour")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["sales"],  mode="lines+markers", name="Sales"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["orders"], mode="lines+markers", name="Orders", yaxis="y2"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["ads"],    mode="lines+markers", name="Ads",    yaxis="y3"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="Ads",    overlaying="y", side="right", position=1.0),
        legend=dict(orientation="h", y=-0.3),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Time series table")
    st.dataframe(ch_df[["hour","sales","orders","ads","ads_ro","sale_ro_row"]].round(3),
                 use_container_width=True)
