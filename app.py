# app.py
# Shopee ROAS Dashboard — overview • channel • compare
# Secrets: ROAS_CSV_URL = "https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>"

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# =============================================================================
# Helpers: load + parse
# =============================================================================
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", str(hdr).strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        return pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

def parse_metrics_cell(s: str):
    if not isinstance(s, str): return [np.nan]*6
    if not re.search(r"\d", s): return [np.nan]*6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p!=""]
    nums = []
    for p in parts[:6]:
        try: nums.append(float(p))
        except: nums.append(np.nan)
    while len(nums)<6: nums.append(np.nan)
    return nums

@st.cache_data(ttl=600, show_spinner=False)
def fetch_csv_text():
    url = os.environ.get("ROAS_CSV_URL", "")
    if not url:
        raise RuntimeError("Missing Secrets: ROAS_CSV_URL")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=600, show_spinner=True)
def load_wide_df():
    return pd.read_csv(io.StringIO(fetch_csv_text()))

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower()=="name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="raw")
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))

    V = pd.DataFrame(df_melt["raw"].apply(parse_metrics_cell).tolist(),
                     columns=["v0","v1","v2","v3","v4","v5"])
    out = pd.concat([df_melt[["timestamp"]+id_cols], V], axis=1)
    out.rename(columns={"name":"channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Mapping ตามไฟล์ของคุณ: v0..v5 -> sales, orders, ads, view, ads_ro, misc
    out["sales"]  = pd.to_numeric(out["v0"], errors="coerce")
    out["orders"] = pd.to_numeric(out["v1"], errors="coerce")
    out["ads"]    = pd.to_numeric(out["v2"], errors="coerce")
    out["view"]   = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro"] = pd.to_numeric(out["v4"], errors="coerce")
    out["misc"]   = pd.to_numeric(out["v5"], errors="coerce")

    # ROAS รวมจาก sales/ads
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty: return df
    tz = str(df["timestamp"].dt.tz)
    target_hour = at_ts.tz_convert(tz).floor("H")
    snap = (df[df["timestamp"].dt.floor("H")==target_hour]
            .sort_values("timestamp").groupby("channel").tail(1))
    return snap

def current_and_yesterday_snapshots(df: pd.DataFrame):
    if df.empty: return df, df, pd.NaT
    cur_ts = df["timestamp"].max()
    return pick_snapshot_at(df, cur_ts), pick_snapshot_at(df, cur_ts-pd.Timedelta(days=1)), cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales"].sum()
    orders= snap["orders"].sum()
    ads   = snap["ads"].sum()
    sale_ro = (sales/ads) if ads!=0 else np.nan
    ads_ro_avg = snap.loc[snap["ads_ro"]>0,"ads_ro"].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0,None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr): return None
    return (curr-prev)*100.0/prev

# =============================================================================
# Common: build hourly-latest dataframe & overlay-by-day helper
# =============================================================================
METRIC_MAP = {
    "sales":"sales",
    "orders":"orders",
    "ads":"ads",
    "sale_ro":"SaleRO",   # ชื่อคอลัมน์จริง
    "ads_ro":"ads_ro",    # ชื่อคอลัมน์จริง
}

def build_hourly_latest(df: pd.DataFrame) -> pd.DataFrame:
    """เลือกสแนปช็อตล่าสุดของแต่ละ 'ชั่วโมง-ช่อง' """
    tmp = df.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    idx = tmp.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return tmp.loc[idx].copy()

def build_overlay_by_day(df_in: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    คืน df => columns: day, hhmm, value (พร้อม clip เฉพาะสำหรับ plot)
    - สำหรับ sales/orders/ads ใช้ sum
    - สำหรับ SaleRO/ads_ro ใช้ mean
    """
    col = METRIC_MAP[metric_key]          # ชื่อคอลัมน์จริง
    hourly = build_hourly_latest(df_in)   # ให้เส้นไม่โยงมั่ว

    ov = hourly[["hour", col]].copy()
    ov["day"] = ov["hour"].dt.date
    ov["hhmm"] = ov["hour"].dt.strftime("%H:%M")

    agg = "sum" if col in ["sales","orders","ads"] else "mean"
    ov = ov.groupby(["day","hhmm"], as_index=False).agg(value=(col, agg))

    # clip เฉพาะตอน plot สำหรับ ROAS/ads_ro
    if col in ["SaleRO","ads_ro"]:
        ov["value"] = ov["value"].clip(upper=50)

    return ov

def overlay_line_fig(ov: pd.DataFrame, y_label: str):
    """วาดกราฟซ้อนตามวัน (x = HH:MM)"""
    fig = go.Figure()
    for day, g in ov.groupby("day"):
        fig.add_trace(go.Scatter(
            x=g["hhmm"], y=g["value"], mode="lines+markers", name=str(day)
        ))
    fig.update_layout(
        height=380,
        xaxis_title="Time (HH:MM)",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified"
    )
    return fig

# =============================================================================
# UI
# =============================================================================
st.sidebar.header("Filters")
if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear(); load_wide_df.clear(); build_long.clear()
    st.experimental_rerun()

try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.dataframe(wide.head() if 'wide' in locals() else pd.DataFrame())
    st.stop()

tz = "Asia/Bangkok"
now_ts = pd.Timestamp.now(tz=tz)

min_ts = df_long["timestamp"].min()
max_ts = df_long["timestamp"].max()
date_max = max_ts.date()
date_min_default = (max_ts - pd.Timedelta(days=2)).date()

d1, d2 = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(date_min_default, date_max),
    min_value=min_ts.date(),
    max_value=date_max,
)
if isinstance(d1,(list,tuple)): d1,d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
selected_channels = all_channels if ("[All]" in chosen or not any(c in all_channels for c in chosen)) \
                    else [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview","Channel","Compare"])

mask = (
    (df_long["timestamp"]>=start_ts) &
    (df_long["timestamp"]<=end_ts) &
    (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Overview
# =============================================================================
if page=="Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period."); st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d)
    cur = kpis_from_snapshot(cur_snap); prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}", delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}", delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}", delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    C[3].metric("sale_ro (Sales/Ads)", "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    C[4].metric("ads_ro (avg>0)", "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # ---- Trend overlay by day (เลือกได้ 1 metric) ----
    st.markdown("### Trend overlay by day")
    metric_choice = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                                 options=["sales","orders","ads","sale_ro","ads_ro"],
                                 index=0, key="ov_metric")
    try:
        ov = build_overlay_by_day(d, metric_choice)
        fig = overlay_line_fig(ov, y_label=metric_choice)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plot failed: {e}")

    # ---- Heatmap prime hours (ยังใช้ sales เช่นเดิม) ----
    st.markdown("### Prime hours heatmap")
    hourly = build_hourly_latest(d)
    hm = hourly.copy()
    hm["day"] = hm["hour"].dt.date
    hm["h"]   = hm["hour"].dt.hour
    heat = hm.groupby(["day","h"]).agg(val=("sales","sum")).reset_index()
    pivot = heat.pivot(index="day", columns="h", values="val").sort_index(ascending=False)
    st.plotly_chart(px.imshow(pivot, aspect="auto", labels=dict(x="Hour", y="Day", color="Sales")),
                    use_container_width=True)

    # (ปล่อยตารางดิบไว้ตามเดิม; คุณบอกว่าจะไปแก้ขั้นถัดไป)
    st.markdown("### Data (hourly latest snapshot per channel)")
    show = hourly[["hour","channel","ads","orders","sales","SaleRO","ads_ro"]].rename(
        columns={"ads":"budget(ads)"}
    ).sort_values(["hour","channel"])
    st.dataframe(show.round(3), use_container_width=True, height=360)

# =============================================================================
# Channel
# =============================================================================
elif page=="Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = df_long[(df_long["channel"]==ch) &
                    (df_long["timestamp"]>=start_ts) &
                    (df_long["timestamp"]<=end_ts)].copy()
    if ch_df.empty:
        st.warning("No data for this channel in selected period."); st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df)
    cur = kpis_from_snapshot(cur_snap); prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}", delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}", delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}", delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    C[3].metric("sale_ro (Sales/Ads)", "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    C[4].metric("ads_ro (avg>0)", "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    st.markdown("### Trend overlay by day (channel)")
    metric_choice = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                                 options=["sales","orders","ads","sale_ro","ads_ro"],
                                 index=0, key="ch_metric")
    try:
        ov = build_overlay_by_day(ch_df, metric_choice)
        fig = overlay_line_fig(ov, y_label=metric_choice)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plot failed: {e}")

    # ตาราง time series ของช่องเดียว (hourly latest)
    st.markdown("### Time series table")
    ch_hourly = build_hourly_latest(ch_df).sort_values("hour")
    st.dataframe(
        ch_hourly[["hour","ads","orders","sales","SaleRO","ads_ro"]]
        .rename(columns={"ads":"budget(ads)"}).round(3),
        use_container_width=True, height=380
    )

# =============================================================================
# Compare (เหมือนเดิม)
# =============================================================================
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels,
                          default=all_channels[:2], max_selections=4)
    if len(pick)<2:
        st.info("Please pick at least 2 channels."); st.stop()
    sub = df_long[(df_long["channel"].isin(pick)) &
                  (df_long["timestamp"]>=start_ts) &
                  (df_long["timestamp"]<=end_ts)].copy()
    if sub.empty:
        st.warning("No data for selected channels in range."); st.stop()

    hourly = build_hourly_latest(sub)
    st.subheader(f"Compare: {', '.join(pick)}")

    kpis = hourly.groupby("channel").agg(
        ROAS=("SaleRO","mean"),
        AOV=("sales", lambda s: (s.sum()/hourly.loc[s.index,"orders"].sum()) if hourly.loc[s.index,"orders"].sum() else np.nan),
        CPO=("orders", lambda s: (hourly.loc[s.index,"ads"].sum()/s.sum()) if s.sum() else np.nan),
        RPV=("sales", lambda s: (s.sum()/hourly.loc[s.index,"view"].sum()) if hourly.loc[s.index,"view"].sum() else np.nan),
        ORV=("orders", lambda s: (s.sum()/hourly.loc[s.index,"view"].sum()) if hourly.loc[s.index,"view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS","sales","orders","ads"], index=0)
    piv = hourly.pivot_table(index="hour", columns="channel",
                             values=("SaleRO" if met=="ROAS" else met), aggfunc="sum").sort_index()
    rel = (piv.div(piv[base], axis=0)-1.0)*100.0

    fig = go.Figure()
    for c in rel.columns:
        if c==base: continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], mode="lines+markers", name=f"{c} vs {base}"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference",
                      legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = hourly.pivot_table(index="hour", columns="channel", values="SaleRO", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], mode="lines", name=c))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
