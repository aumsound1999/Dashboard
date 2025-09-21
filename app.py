# app.py
# Shopee ROAS Dashboard — Overview • Channel • Compare
# ----------------------------------------------------

import os, re, io, math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date, timezone, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests

# =========================
# App setup
# =========================
st.set_page_config(page_title="Shopee ROAS", layout="wide")
st.title("Shopee ROAS Dashboard")

# -------- Auto reload every 10 minutes (600,000 ms)
try:
    from streamlit_autorefresh import st_autorefresh  # optional
    st_autorefresh(interval=600_000, key="auto")
except Exception:
    st.markdown(
        "<script>setTimeout(function(){window.location.reload();},600000);</script>",
        unsafe_allow_html=True,
    )

# =========================
# Data loader (from Secrets)
# =========================
CSV_URL = os.environ.get("ROAS_CSV_URL", "").strip()

def _fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

@st.cache_data(ttl=600, show_spinner=False)  # cache 10 minutes
def get_wide_df(url: str) -> pd.DataFrame:
    return _fetch_csv(url)

def load_or_reload(force=False):
    if not CSV_URL:
        st.error("Missing ROAS_CSV_URL in Secrets.")
        st.stop()
    if force:
        get_wide_df.clear()
    df = get_wide_df(CSV_URL)
    st.session_state["df_wide"] = df
    st.session_state["last_refresh"] = datetime.now(timezone.utc)
    return df

# initial load (first visit)
df_wide = st.session_state.get("df_wide")
if df_wide is None:
    df_wide = load_or_reload(force=False)

# =========================
# Helpers for shaping / KPIs
# =========================

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:4

def is_time_col(col: str) -> bool:
    return TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, year=None, month=None, tz="Asia/Bangkok") -> pd.Timestamp:
    # hdr example: "D21 12:4" -> day=21, hour=12, minute=4
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
    # metrics in a cell like "budget,user,order,view,sale,ro,(maybe extra...)".
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
    cols = list(df_wide.columns)
    id_cols, time_cols = [], []
    for c in cols:
        if c.strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
        else:
            pass  # ignore meta
    if not time_cols:
        raise ValueError("No time columns detected (e.g. 'D21 12:4').")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")
    ts = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))
    df_melt["timestamp"] = ts

    metrics = df_melt["metrics"].apply(str_to_metrics_tuple)
    metrics_df = pd.DataFrame(metrics.tolist(),
                              columns=["budget","user","order","view","sale","ro"])
    out = pd.concat([df_melt[["timestamp"] + id_cols], metrics_df], axis=1)
    out.rename(columns={"name":"channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    for c in ["budget","user","order","view","sale","ro"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # KPIs
    out["ROAS"] = out["sale"] / out["budget"].replace(0, np.nan)
    out["AOV"]  = out["sale"] / out["order"].replace(0, np.nan)
    out["CPO"]  = out["budget"] / out["order"].replace(0, np.nan)
    out["RPV"]  = out["sale"] / out["view"].replace(0, np.nan)
    out["ORV"]  = out["order"] / out["view"].replace(0, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df):
    return long_from_wide(df)

def aggregate_hourly(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("H")
    idx = df.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return df.loc[idx]

def compute_totals(df):
    if df is None or df.empty:
        return dict(Sales=np.nan, Orders=np.nan, Budget=np.nan,
                    ROAS=np.nan, AOV=np.nan, CPO=np.nan, RPV=np.nan, ORV=np.nan)
    totals = {}
    totals["Sales"]  = df["sale"].sum()
    totals["Orders"] = df["order"].sum()
    totals["Budget"] = df["budget"].sum()
    totals["ROAS"]   = (df["sale"].sum() / df["budget"].sum()) if df["budget"].sum()!=0 else np.nan
    totals["AOV"]    = (df["sale"].sum() / df["order"].sum())  if df["order"].sum()!=0  else np.nan
    totals["CPO"]    = (df["budget"].sum() / df["order"].sum()) if df["order"].sum()!=0 else np.nan
    totals["RPV"]    = (df["sale"].sum() / df["view"].sum())   if df["view"].sum()!=0   else np.nan
    totals["ORV"]    = (df["order"].sum() / df["view"].sum())  if df["view"].sum()!=0   else np.nan
    return totals

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr): return None
    return (curr - prev) * 100.0 / prev

def format_big(x):
    if pd.isna(x): return "-"
    try:
        return f"{x:,.0f}"
    except:
        return str(x)

def kpi_card(label, value, delta=None, col=None):
    target = st if col is None else col
    if delta is None:
        target.metric(label, format_big(value))
    else:
        target.metric(label, format_big(value), delta=f"{delta:+.1f}%")

# =========================
# Sidebar: Data source + Filters
# =========================

st.sidebar.header("Data source")
colL, colR = st.sidebar.columns([1,1])
if colL.button("Reload", use_container_width=True):
    df_wide = load_or_reload(force=True)
with colR:
    ts = st.session_state.get("last_refresh")
    st.caption("Last refresh: " + (ts.astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else "-"))

st.sidebar.divider()
st.sidebar.header("Filters")

# Build long dataframe
try:
    df_long = build_long(df_wide)
except Exception as e:
    st.error(f"Failed to parse sheet: {e}")
    st.write(df_wide.head())
    st.stop()

ALL_CHANNELS = sorted(df_long["channel"].dropna().unique().tolist())

# Select all toggle
select_all = st.sidebar.checkbox("Select all channels", value=True)
if select_all:
    selected_channels = ALL_CHANNELS[:]
else:
    selected_channels = st.sidebar.multiselect("Channels", options=ALL_CHANNELS,
                                               default=ALL_CHANNELS[:10])

# Date range from data
tz = "Asia/Bangkok"
min_ts = df_long["timestamp"].min()
max_ts = df_long["timestamp"].max()
if pd.isna(min_ts) or pd.isna(max_ts):
    st.warning("No valid timestamps found."); st.stop()

date_min = min_ts.date()
date_max = max_ts.date()
d1, d2 = st.sidebar.date_input("Date range",
                               value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)
if isinstance(d1, (list, tuple)):  # safety
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

page = st.sidebar.radio("Page", ["Overview","Channel","Compare"])

# =========================
# Filtered datasets
# =========================

# For Overview -> filter by selected channels + date
mask_overview = (df_long["channel"].isin(selected_channels)) & \
                (df_long["timestamp"]>=start_ts) & (df_long["timestamp"]<=end_ts)
d_overview = df_long.loc[mask_overview].copy()

# For Channel/Compare -> use all channels for dropdown, but time-limited
mask_time_only = (df_long["timestamp"]>=start_ts) & (df_long["timestamp"]<=end_ts)
d_time_only = df_long.loc[mask_time_only].copy()

# Snapshot hourly
hourly_overview = aggregate_hourly(d_overview) if not d_overview.empty else d_overview
hourly_time_only = aggregate_hourly(d_time_only) if not d_time_only.empty else d_time_only

# Baseline yesterday (for deltas)
y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts - pd.Timedelta(days=1)
base_mask_overview = (df_long["channel"].isin(selected_channels)) & \
                     (df_long["timestamp"]>=y_start) & (df_long["timestamp"]<=y_end)
baseline_hourly_overview = aggregate_hourly(df_long.loc[base_mask_overview]) \
                           if not df_long.loc[base_mask_overview].empty else None

baseline_time_only = aggregate_hourly(
    df_long.loc[(df_long["timestamp"]>=y_start) & (df_long["timestamp"]<=y_end)]
) if not df_long.loc[(df_long["timestamp"]>=y_start) & (df_long["timestamp"]<=y_end)].empty else None

# =========================
# PAGES
# =========================

if page == "Overview":
    st.subheader("Overview (All selected channels)")

    totals = compute_totals(hourly_overview)
    base_totals = compute_totals(baseline_hourly_overview) if baseline_hourly_overview is not None else {}

    cols = st.columns(8)
    for i,lbl in enumerate(["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]):
        curr = totals.get(lbl, np.nan)
        prev = base_totals.get(lbl, np.nan) if base_totals else np.nan
        delta = pct_delta(curr, prev) if not pd.isna(prev) else None
        kpi_card(lbl, curr, delta=delta, col=cols[i])

    # Trend
    st.markdown("### Trend by hour")
    if not hourly_overview.empty:
        trend = hourly_overview.groupby("hour").agg({"sale":"sum","order":"sum","budget":"sum"}).reset_index()
        trend["ROAS"] = trend["sale"] / trend["budget"].replace(0, np.nan)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["hour"], y=trend["sale"],   mode="lines+markers", name="Sales"))
        fig.add_trace(go.Scatter(x=trend["hour"], y=trend["order"],  mode="lines+markers", name="Orders", yaxis="y2"))
        fig.add_trace(go.Scatter(x=trend["hour"], y=trend["ROAS"],   mode="lines+markers", name="ROAS",  yaxis="y3"))
        fig.update_layout(
            xaxis_title="Hour",
            yaxis=dict(title="Sales"),
            yaxis2=dict(title="Orders", overlaying="y", side="right"),
            yaxis3=dict(title="ROAS", anchor="free", overlaying="y", side="right", position=1.0),
            legend=dict(orientation="h", y=-0.2),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data in selected range.")

    # Heatmap
    st.markdown("### Prime hours heatmap")
    metric = st.selectbox("Heatmap metric", ["Sales","ROAS","ORV"], index=0, key="heat_metric")
    if not hourly_overview.empty:
        tmp = hourly_overview.copy()
        tmp["day"] = tmp["hour"].dt.date
        tmp["hr"]  = tmp["hour"].dt.hour
        agg_map = {"Sales":"sale","ROAS":"ROAS","ORV":"ORV"}
        heat = tmp.groupby(["day","hr"]).agg(val=(agg_map[metric],"mean")).reset_index()
        heat_pivot = heat.pivot(index="day", columns="hr", values="val").sort_index(ascending=False)
        fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                          labels=dict(x="Hour", y="Day", color=metric))
        st.plotly_chart(fig_h, use_container_width=True)

    # Leaderboard
    st.markdown("### Leaderboard by channel")
    if not hourly_overview.empty:
        ldb = hourly_overview.groupby("channel").agg(
            Sales=("sale","sum"), Orders=("order","sum"), Budget=("budget","sum")
        ).reset_index()
        ldb["ROAS"] = ldb["Sales"] / ldb["Budget"].replace(0, np.nan)
        ldb["CPO"]  = ldb["Budget"] / ldb["Orders"].replace(0, np.nan)
        st.dataframe(ldb.round(3).sort_values("ROAS", ascending=False),
                     use_container_width=True)

    # Raw table
    st.markdown("### Data (hourly latest snapshot per channel)")
    if not hourly_overview.empty:
        show_cols = ["hour","channel","budget","user","order","view","sale","ROAS","AOV","CPO","RPV","ORV"]
        st.dataframe(hourly_overview[show_cols].sort_values(["hour","channel"]).round(3),
                     use_container_width=True)

# ---------------- CHANNEL ----------------
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=ALL_CHANNELS, index=0)

    if hourly_time_only.empty:
        st.info("No data in selected range."); st.stop()

    ch_df = hourly_time_only[hourly_time_only["channel"]==ch].copy()
    st.subheader(f"Channel • {ch}")

    # KPI row
    base_df = baseline_time_only[baseline_time_only["channel"]==ch] if baseline_time_only is not None else None
    totals = compute_totals(ch_df)
    base_totals = compute_totals(base_df) if base_df is not None and not base_df.empty else {}
    cols = st.columns(8)
    for i,lbl in enumerate(["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]):
        curr = totals.get(lbl, np.nan)
        prev = base_totals.get(lbl, np.nan) if base_totals else np.nan
        delta = pct_delta(curr, prev) if not pd.isna(prev) else None
        kpi_card(lbl, curr, delta=delta, col=cols[i])

    # Multi-axis line
    st.markdown("### Multi-axis line")
    if not ch_df.empty:
        tr = ch_df.sort_values("hour")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr["hour"], y=tr["sale"],   mode="lines+markers", name="Sales"))
        fig.add_trace(go.Scatter(x=tr["hour"], y=tr["order"],  mode="lines+markers", name="Orders", yaxis="y2"))
        fig.add_trace(go.Scatter(x=tr["hour"], y=tr["budget"], mode="lines+markers", name="Budget", yaxis="y3"))
        fig.add_trace(go.Scatter(x=tr["hour"], y=tr["ROAS"],   mode="lines+markers", name="ROAS", yaxis="y4"))
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

        # 24h distribution
        st.markdown("### 24h distribution")
        tr["h"] = tr["hour"].dt.hour
        dbar = tr.groupby("h").agg(Sales=("sale","sum"), ORV=("ORV","mean"), RPV=("RPV","mean")).reset_index()
        st.plotly_chart(px.bar(dbar, x="h", y="Sales"), use_container_width=True)
        st.caption("Bars show Sales/hour; hover to see ORV/RPV (mean).")
    else:
        st.info("This channel has no data in the selected range.")

# ---------------- COMPARE ----------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=ALL_CHANNELS, default=ALL_CHANNELS[:2])
    if len(pick) < 2:
        st.info("Please pick at least 2 channels."); st.stop()
    if len(pick) > 4:
        pick = pick[:4]
    st.subheader(f"Compare: {', '.join(pick)}")

    if hourly_time_only.empty:
        st.info("No data in selected range."); st.stop()

    sub = hourly_time_only[hourly_time_only["channel"].isin(pick)].copy()

    # KPI comparison table
    st.markdown("### KPI comparison table")
    kpis = sub.groupby("channel").agg(
        ROAS=("ROAS","mean"), AOV=("AOV","mean"),
        CPO=("CPO","mean"), RPV=("RPV","mean"), ORV=("ORV","mean")
    ).reset_index()
    st.dataframe(kpis.round(3), use_container_width=True)

    # Diff vs baseline channel
    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS","sale","order","budget"], index=0)
    pivot = sub.pivot_table(index="hour", columns="channel", values=met, aggfunc="sum").sort_index()
    if base in pivot.columns:
        rel = (pivot.div(pivot[base], axis=0)-1.0)*100.0
        fig = go.Figure()
        for c in rel.columns:
            if c == base: 
                continue
            fig.add_trace(go.Scatter(x=rel.index, y=rel[c], mode="lines+markers", name=f"{c} vs {base}"))
        fig.update_layout(yaxis_title="% difference", xaxis_title="Hour",
                          height=420, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Baseline not in data.")

    st.markdown("### Small multiples (ROAS)")
    sm = sub.pivot_table(index="hour", columns="channel", values="ROAS", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], mode="lines", name=c))
    fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig2, use_container_width=True)
