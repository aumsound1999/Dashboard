# app.py
# -----------------------------------------------------------
# Shopee ROAS Dashboard (Overview • Channel)
# - Multi-metric selector (Sales, Orders, Budget(Ads), sale_ro, ads_ro)
# - Overlay-by-day charts (lines per day on same time axis)
# - Hourly latest snapshot logic
# - %Δ vs yesterday
# - Default date range = last 3 days
# - Reload button + auto cache 10 minutes
# -----------------------------------------------------------

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------
# Parse helpers (same as before)
# -----------------------------
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:4

def is_time_col(col: str) -> bool:
    return TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, year=None, month=None, tz="Asia/Bangkok") -> pd.Timestamp:
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr.strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    if year is None:
        year = now.year
    if month is None:
        month = now.month
    try:
        ts = pd.Timestamp(year=year, month=month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        ts = pd.NaT
    return ts

def str_to_metrics_tuple(s: str):
    # Expect cell like: "2314,17,47,1400,29,46,1024"
    # Keep first 6 numbers: budget,user,order,view,sale,ro
    if not isinstance(s, str):
        return (np.nan,)*6
    if not re.search(r"\d", s):
        return (np.nan,)*6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]
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
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if c.strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")

    # Parse timestamp from col header
    ts = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))
    df_melt["timestamp"] = ts

    # Parse metrics
    metrics = df_melt["metrics"].apply(str_to_metrics_tuple)
    metrics_df = pd.DataFrame(metrics.tolist(),
                              columns=["budget","user","order","view","sale","ro"])

    out = pd.concat([df_melt[["timestamp"] + id_cols], metrics_df], axis=1)
    out.rename(columns={"name": "channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # numeric
    for c in ["budget","user","order","view","sale","ro"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Derived KPIs (row-level)
    out["ROAS"] = out["sale"] / out["budget"].replace(0, np.nan)
    out["AOV"]  = out["sale"] / out["order"].replace(0, np.nan)
    out["CPO"]  = out["budget"] / out["order"].replace(0, np.nan)
    out["RPV"]  = out["sale"] / out["view"].replace(0, np.nan)
    out["ORV"]  = out["order"] / out["view"].replace(0, np.nan)
    return out

# -----------------------------
# Data loading / caching
# -----------------------------
@st.cache_data(ttl=60*10, show_spinner=False)
def load_sheet():
    url = os.environ.get("ROAS_CSV_URL", "")
    if not url:
        raise RuntimeError("Missing ROAS_CSV_URL. Please set secret in Streamlit.")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

@st.cache_data(ttl=60*10, show_spinner=False)
def get_long():
    df_wide = load_sheet()
    return long_from_wide(df_wide)

def reload_data():
    load_sheet.clear()
    get_long.clear()

# -----------------------------
# Snapshot logic / aggregations
# -----------------------------
def latest_hourly_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep latest snapshot per (channel, hour)."""
    d = df.copy()
    d["hour"] = d["timestamp"].dt.floor("H")
    idx = d.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return d.loc[idx].copy()

def default_dates(df: pd.DataFrame, days=3):
    """Default date range = last `days` with data."""
    if df["timestamp"].empty:
        today = pd.Timestamp.now(tz="Asia/Bangkok").date()
        return today - timedelta(days=days-1), today
    max_ts = df["timestamp"].max()
    end_date = max_ts.date()
    start_date = end_date - timedelta(days=days-1)
    return start_date, end_date

def compute_overview_agg(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across selected channels per hour.
    For sale_ro: total sale / total budget per hour.
    For ads_ro: mean ROAS across channels with ROAS>0 per hour.
    """
    d = hourly.copy()
    # sum metrics per hour
    grp = d.groupby("hour", as_index=False).agg(
        sale=("sale","sum"),
        order=("order","sum"),
        budget=("budget","sum")
    )

    # sale_ro at hour: total sale / total budget
    grp["sale_ro"] = grp["sale"] / grp["budget"].replace(0, np.nan)

    # ads_ro: mean ROAS across channels with ROAS>0
    g2 = d[d["ROAS"]>0].groupby("hour")["ROAS"].mean().rename("ads_ro")
    grp = grp.merge(g2, on="hour", how="left")

    # add date-of-day to support overlay
    grp["day"] = grp["hour"].dt.date
    grp["time"] = grp["hour"].dt.time
    grp["hhmm"] = grp["hour"].dt.strftime("%H:%M")
    return grp

def compute_channel_agg(hourly: pd.DataFrame, ch: str) -> pd.DataFrame:
    """
    For a single channel; sale_ro becomes sale/budget at hour;
    ads_ro = this channel's ROAS.
    """
    d = hourly[hourly["channel"]==ch].copy()
    d["sale_ro"] = d["sale"] / d["budget"].replace(0, np.nan)
    d["ads_ro"]  = d["ROAS"]
    d["day"]     = d["hour"].dt.date
    d["hhmm"]    = d["hour"].dt.strftime("%H:%M")
    return d[["hour","day","hhmm","sale","order","budget","sale_ro","ads_ro"]].reset_index(drop=True)

def kpi_totals(df_hours: pd.DataFrame) -> dict:
    """
    KPI totals for top row (across the selected time window)
    """
    out = {}
    S = df_hours["sale"].sum()
    O = df_hours["order"].sum()
    B = df_hours["budget"].sum()
    out["Sales"]  = S
    out["Orders"] = O
    out["Budget"] = B
    out["ROAS"]   = S / B if B else np.nan
    out["AOV"]    = S / O if O else np.nan
    out["CPO"]    = B / O if O else np.nan

    # mean across hours
    out["RPV"]    = (df_hours["sale"].sum() / df_hours["order"].sum()) if df_hours["order"].sum() else np.nan
    out["ORV"]    = (df_hours["order"].sum() / df_hours["view"].sum()) if "view" in df_hours.columns and df_hours["view"].sum() else np.nan
    return out

def pct_delta(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    if curr is None or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

def format_big(x):
    if pd.isna(x): 
        return "-"
    try:
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:,.2f}"
    except:
        return str(x)

def kpi_row(tot_curr: dict, tot_prev: dict, cols, labels):
    for i,l in enumerate(labels):
        curr = tot_curr.get(l, np.nan)
        prev = tot_prev.get(l, np.nan) if tot_prev else np.nan
        delta = pct_delta(curr, prev)
        arrow = ""
        if delta is not None:
            arrow = f"{delta:+.1f}%"
        cols[i].metric(l, format_big(curr), arrow)

# -----------------------------
# Plot helpers (overlay by day)
# -----------------------------
METRIC_LABELS = {
    "sale":"Sales",
    "order":"Orders",
    "budget":"Budget(Ads)",
    "sale_ro":"sale_ro (Total Sales / Total Ads)",
    "ads_ro":"ads_ro (Avg ROAS of channels>0)"
}

def plot_overlay_by_day(df_agg: pd.DataFrame, metric: str, title: str):
    """
    df_agg must have columns: ['day', 'hhmm', metric]
    One trace per day on the same HH:MM axis
    """
    fig = go.Figure()
    for d, sdf in df_agg.groupby("day"):
        sdf = sdf.sort_values("hhmm")
        fig.add_trace(
            go.Scatter(
                x=sdf["hhmm"], y=sdf[metric],
                mode="lines+markers", name=str(d)
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=20,r=20,t=40,b=20),
        xaxis_title="Time (HH:MM)",
        yaxis_title=METRIC_LABELS.get(metric, metric),
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# UI
# -----------------------------
st.title("Shopee ROAS Dashboard")

with st.sidebar:
    st.markdown("### Filters")

    # Load & parse
    long_df = get_long()

    # Default date window = last 3 days based on data
    _d1, _d2 = default_dates(long_df, days=3)
    d1, d2 = st.date_input(
        "Date range (default 3 days)",
        value=(_d1, _d2),
        min_value=long_df["timestamp"].min().date(),
        max_value=long_df["timestamp"].max().date()
    )
    if isinstance(d1, (list,tuple)):  # streamlit older quirk
        d1, d2 = d1

    # Channel filter (multi) + All tag
    all_channels = sorted(long_df["channel"].dropna().unique().tolist())
    st.markdown("Channels (เลือก All ได้)")
    pick = st.multiselect("",
                          options=["[All]"]+all_channels,
                          default=["[All]"],
                          label_visibility="collapsed")
    if "[All]" in pick:
        sel_channels = all_channels
    else:
        sel_channels = pick

    # Page selector
    page = st.radio("Page", ["Overview","Channel"], index=0)

# Reload button
c1, c2 = st.columns([1,5])
with c1:
    if st.button("Reload", type="secondary"):
        reload_data()
        st.experimental_rerun()
with c2:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Filter long_df with date+channels
tz = "Asia/Bangkok"
start_ts = pd.Timestamp.combine(pd.to_datetime(d1).date(), pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(pd.to_datetime(d2).date(), pd.Timestamp("23:59").time()).tz_localize(tz)

mask = (long_df["channel"].isin(sel_channels)) & (long_df["timestamp"]>=start_ts) & (long_df["timestamp"]<=end_ts)
df_sel = long_df.loc[mask].copy()

if df_sel.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# hourly snapshot
hourly = latest_hourly_snapshot(df_sel)

# yesterday baseline window (same hour range)
y_start = start_ts - timedelta(days=1)
y_end   = end_ts - timedelta(days=1)
base_mask = (long_df["channel"].isin(sel_channels)) & (long_df["timestamp"]>=y_start) & (long_df["timestamp"]<=y_end)
baseline_hourly = latest_hourly_snapshot(long_df.loc[base_mask]) if not long_df.loc[base_mask].empty else pd.DataFrame()

# -----------------------------
# Top KPIs (+ %Δ vs yesterday)
# -----------------------------
st.subheader("Overview (All selected channels)" if page=="Overview" else "Channel")

if page == "Overview":
    # aggregate hours across channels
    agg = compute_overview_agg(hourly)
    prev_agg = compute_overview_agg(baseline_hourly) if not baseline_hourly.empty else None

    # KPI current totals across hours
    tot_curr = {
        "Sales": agg["sale"].sum(),
        "Orders": agg["order"].sum(),
        "Budget": agg["budget"].sum(),
        "ROAS": (agg["sale"].sum()/agg["budget"].sum() if agg["budget"].sum() else np.nan),
        "AOV": (agg["sale"].sum()/agg["order"].sum() if agg["order"].sum() else np.nan),
        "CPO": (agg["budget"].sum()/agg["order"].sum() if agg["order"].sum() else np.nan),
        "RPV": np.nan,  # not used here (requires 'view' summed)
        "ORV": np.nan
    }
    tot_prev = None
    if prev_agg is not None and not prev_agg.empty:
        tot_prev = {
            "Sales": prev_agg["sale"].sum(),
            "Orders": prev_agg["order"].sum(),
            "Budget": prev_agg["budget"].sum(),
            "ROAS": (prev_agg["sale"].sum()/prev_agg["budget"].sum() if prev_agg["budget"].sum() else np.nan),
            "AOV": (prev_agg["sale"].sum()/prev_agg["order"].sum() if prev_agg["order"].sum() else np.nan),
            "CPO": (prev_agg["budget"].sum()/prev_agg["order"].sum() if prev_agg["order"].sum() else np.nan),
            "RPV": np.nan,
            "ORV": np.nan
        }

    cols = st.columns(8)
    labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
    kpi_row(tot_curr, tot_prev, cols, labels)

    # ---- Metric selector & overlay charts ----
    st.markdown("### Trend overlay by day")
    metric_pick = st.multiselect(
        "Metrics to plot",
        options=[("sale","Sales"),("order","Orders"),("budget","Budget(Ads)"),
                 ("sale_ro","sale_ro (Total Sales/Total Ads)"),
                 ("ads_ro","ads_ro (Avg ROAS of channels>0)")],
        default=[("sale","Sales"),("order","Orders"),("budget","Budget(Ads)")],
        format_func=lambda x: x[1]
    )
    chosen_codes = [m[0] for m in metric_pick]
    if not chosen_codes:
        st.info("Please pick at least 1 metric.")
        st.stop()

    # Plot each metric as its own chart (อ่านง่ายสุด)
    for code in chosen_codes:
        plot_overlay_by_day(agg[["day","hhmm",code]].dropna(subset=[code]), code, METRIC_LABELS.get(code, code))

    # Raw hourly table (optional)
    st.markdown("### Data (hourly latest snapshot aggregated)")
    st.dataframe(agg.sort_values(["day","hhmm"]), use_container_width=True)

else:
    # ---- Channel page ----
    # Picker should list ALL channels in sheet (แม้ฟิลเตอร์ด้านซ้ายไม่เลือกช่องก็เลือกได้)
    ch = st.selectbox("Pick one channel", options=sorted(long_df["channel"].dropna().unique().tolist()))
    ch_hourly = latest_hourly_snapshot(df_sel)  # hourly within filter window (date)
    # build channel agg using *hourly of all data for date window*, but limited to selected ch
    ch_df = latest_hourly_snapshot(long_df[(long_df["channel"]==ch) &
                                           (long_df["timestamp"]>=start_ts) &
                                           (long_df["timestamp"]<=end_ts)])
    if ch_df.empty:
        st.warning("No data for this channel in the selected date range.")
        st.stop()

    ch_agg = compute_channel_agg(ch_df, ch)

    # Baseline (yesterday)
    ch_prev_df = latest_hourly_snapshot(long_df[(long_df["channel"]==ch) &
                       (long_df["timestamp"]>=y_start) & (long_df["timestamp"]<=y_end)])
    ch_prev = compute_channel_agg(ch_prev_df, ch) if not ch_prev_df.empty else None

    # top KPIs for this channel (sum across hours)
    tot_curr = {
        "Sales": ch_agg["sale"].sum(),
        "Orders": ch_agg["order"].sum(),
        "Budget": ch_agg["budget"].sum(),
        "ROAS": (ch_agg["sale"].sum()/ch_agg["budget"].sum() if ch_agg["budget"].sum() else np.nan),
        "AOV": (ch_agg["sale"].sum()/ch_agg["order"].sum() if ch_agg["order"].sum() else np.nan),
        "CPO": (ch_agg["budget"].sum()/ch_agg["order"].sum() if ch_agg["order"].sum() else np.nan),
        "RPV": np.nan,
        "ORV": np.nan
    }
    tot_prev = None
    if ch_prev is not None and not ch_prev.empty:
        tot_prev = {
            "Sales": ch_prev["sale"].sum(),
            "Orders": ch_prev["order"].sum(),
            "Budget": ch_prev["budget"].sum(),
            "ROAS": (ch_prev["sale"].sum()/ch_prev["budget"].sum() if ch_prev["budget"].sum() else np.nan),
            "AOV": (ch_prev["sale"].sum()/ch_prev["order"].sum() if ch_prev["order"].sum() else np.nan),
            "CPO": (ch_prev["budget"].sum()/ch_prev["order"].sum() if ch_prev["order"].sum() else np.nan),
            "RPV": np.nan,
            "ORV": np.nan
        }

    cols = st.columns(8)
    labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
    kpi_row(tot_curr, tot_prev, cols, labels)

    # metric selector
    st.markdown(f"### {ch} • overlay by day")
    metric_pick = st.multiselect(
        "Metrics to plot",
        options=[("sale","Sales"),("order","Orders"),("budget","Budget(Ads)"),
                 ("sale_ro","sale_ro (Sales/Ads)"),
                 ("ads_ro","ads_ro (ROAS)")],
        default=[("sale","Sales"),("order","Orders"),("budget","Budget(Ads)")],
        format_func=lambda x: x[1]
    )
    chosen_codes = [m[0] for m in metric_pick]
    if not chosen_codes:
        st.info("Please pick at least 1 metric.")
        st.stop()

    for code in chosen_codes:
        plot_overlay_by_day(ch_agg[["day","hhmm",code]].dropna(subset=[code]), code, METRIC_LABELS.get(code, code))

    st.markdown("### Data (hourly latest snapshot for this channel)")
    st.dataframe(ch_agg.sort_values(["day","hhmm"]), use_container_width=True)
