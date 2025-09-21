# app.py
# Shopee ROAS Dashboard — overview • channel • compare (hourly version)

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# =========================
# Config
# =========================
TZ      = "Asia/Bangkok"
RO_CAP  = 50.0  # cap สำหรับ sale_ro / ads_ro
HOURS_24 = [f"{h:02d}:00" for h in range(24)]  # 00:00..23:00

# =========================
# Parse Google Sheet
# =========================
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = TZ) -> pd.Timestamp:
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

def parse_metrics_cell(s: str):
    if not isinstance(s, str) or not re.search(r"\d", s):
        return [np.nan]*6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]
    out = []
    for p in parts[:6]:
        try: out.append(float(p))
        except: out.append(np.nan)
    while len(out) < 6:
        out.append(np.nan)
    return out

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
    text = fetch_csv_text()
    return pd.read_csv(io.StringIO(text))

def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    melted = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                          var_name="time_col", value_name="raw")
    melted["timestamp"] = melted["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=TZ))

    parsed = melted["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([melted[["timestamp"] + id_cols], V], axis=1)\
            .rename(columns={"name":"channel"})\
            .dropna(subset=["timestamp"]).reset_index(drop=True)

    # mapping ตามที่ยืนยันกัน
    out["sales"]   = pd.to_numeric(out["v0"], errors="coerce")   # สะสม
    out["orders"]  = pd.to_numeric(out["v1"], errors="coerce")   # สะสม
    out["ads"]     = pd.to_numeric(out["v2"], errors="coerce")   # สะสม (ค่าโฆษณา)
    out["view"]    = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro0"] = pd.to_numeric(out["v4"], errors="coerce")   # raw ro เดิม (ไม่ได้ใช้แล้วในกราฟ)
    out["misc"]    = pd.to_numeric(out["v5"], errors="coerce")

    return out

@st.cache_data(ttl=600, show_spinner=False)
def get_long():
    return wide_to_long(load_wide_df())

# =========================
# KPI snapshot เหมือนเดิม
# =========================
def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    target_hour = at_ts.tz_convert(TZ).floor("H")
    snap = (
        df[df["timestamp"].dt.floor("H") == target_hour]
        .sort_values("timestamp")
        .groupby("channel")
        .tail(1)
    )
    return snap

def current_and_yesterday_snapshots(df: pd.DataFrame):
    if df.empty:
        return df, df, pd.NaT
    cur_ts   = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts)
    y_snap   = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1))
    return cur_snap, y_snap, cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales  = snap["sales"].sum()
    orders = snap["orders"].sum()
    ads    = snap["ads"].sum()
    sale_ro = (sales/ads) if ads else np.nan
    ads_ro_avg = snap["ads_ro0"][snap["ads_ro0"]>0].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr):      return None
    return (curr - prev) * 100.0 / prev

# =========================
# Hourly overlay & heatmap
# =========================
def _hourly_latest_agg(d: pd.DataFrame, chs: list) -> pd.DataFrame:
    """สแนปช็อตล่าสุดต่อชั่วโมง/ช่อง แล้วรวมทุกช่องเป็นรายชั่วโมง/วัน"""
    sub = d[d["channel"].isin(chs)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["hour"] = sub["timestamp"].dt.floor("H")
    sub["day"]  = sub["hour"].dt.date
    sub["hh"]   = sub["hour"].dt.strftime("%H:00")

    # เลือกข้อมูลล่าสุดของแต่ละ channel+hour
    latest = sub.sort_values("timestamp").groupby(["channel","hour"]).tail(1)

    # รวมทุกช่องในแต่ละ hour
    agg = latest.groupby(["day","hour","hh"], as_index=False).agg(
        sales=("sales","sum"),
        orders=("orders","sum"),
        ads=("ads","sum"),
    )
    # ads_sales/ads_spend สำหรับ ads_ro
    agg["ads_sales"]  = agg["sales"]
    agg["ads_spend"]  = agg["ads"]
    return agg

def _diff_clip_nonneg(ser: pd.Series, by: pd.Series) -> pd.Series:
    v = ser.groupby(by).diff()
    v = v.clip(lower=0)            # กันค่าติดลบ
    v = v.groupby(by).ffill()       # กันเส้นขาด
    return v

def build_overlay_by_day_hour(d: pd.DataFrame, metric: str, chs: list) -> pd.DataFrame:
    """
    คืน pivot index=hh(00:00..23:00), columns=day, values=metric
    - sales/orders/ads -> Δ ต่อชั่วโมง (ไม่ติดลบ)
    - sale_ro -> (Δsales/Δads) cap 50
    - ads_ro  -> (Δads_sales/Δads_spend) cap 50
    """
    agg = _hourly_latest_agg(d, chs)
    if agg.empty:
        return pd.DataFrame()

    ds = _diff_clip_nonneg(agg["sales"],  agg["day"])
    do = _diff_clip_nonneg(agg["orders"], agg["day"])
    da = _diff_clip_nonneg(agg["ads"],    agg["day"])

    sale_ro = (ds / da.replace(0, np.nan)).clip(upper=RO_CAP)
    sale_ro = sale_ro.groupby(agg["day"]).ffill()

    ads_sales_delta = _diff_clip_nonneg(agg["ads_sales"], agg["day"])
    ads_spend_delta = _diff_clip_nonneg(agg["ads_spend"], agg["day"])
    ads_ro = (ads_sales_delta / ads_spend_delta.replace(0, np.nan)).clip(upper=RO_CAP)
    ads_ro = ads_ro.groupby(agg["day"]).ffill()

    if   metric == "sales" : use = ds
    elif metric == "orders": use = do
    elif metric == "ads"   : use = da
    elif metric == "sale_ro": use = sale_ro
    elif metric == "ads_ro" : use = ads_ro
    else:
        raise ValueError("Unknown metric")

    Z = pd.DataFrame({"hh": agg["hh"], "day": agg["day"], "val": use})
    piv = Z.pivot_table(index="hh", columns="day", values="val", aggfunc="last")
    piv = piv.reindex(HOURS_24)  # ให้ครบ 00..23 ทุกวัน
    return piv

def build_heatmap_hourly(d: pd.DataFrame, metric: str, chs: list) -> pd.DataFrame:
    """คืนตาราง day x hour สำหรับ heatmap (ใช้ค่าเดียวกับ overlay เพื่อให้ตรงกัน)"""
    piv = build_overlay_by_day_hour(d, metric, chs)
    if piv.empty:
        return piv
    return piv.T  # rows=day, cols=hour

# =========================
# UI
# =========================
st.sidebar.header("Filters")
if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear(); load_wide_df.clear(); get_long.clear()
    st.experimental_rerun()

df_long = get_long()
now_ts  = pd.Timestamp.now(tz=TZ)

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
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(TZ)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(TZ)

all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=["[All]"]+all_channels, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview","Channel","Compare"])

mask = (
    (df_long["timestamp"] >= start_ts) &
    (df_long["timestamp"] <= end_ts) &
    (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# Overview
# =========================
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d)
    cur  = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",  f"{pct_delta(cur['Sales'],  prev['Sales']):+.1f}%" if prev['Sales'] else None)
    C[1].metric("Orders",f"{cur['Orders']:,.0f}", f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None)
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}", f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None)
    C[3].metric("sale_ro (Sales/Ads)", "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None)
    C[4].metric("ads_ro (avg>0)", "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None)
    st.caption(f"Snapshot hour: {cur_hour}")

    st.markdown("### Trend overlay by day")
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", options=["sales","orders","ads","sale_ro","ads_ro"], index=0)
    piv = build_overlay_by_day_hour(d, metric, selected_channels)
    if piv.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for day in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[day], name=str(day), mode="lines+markers"))
        fig.update_layout(height=420, xaxis_title="Hour", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prime hours heatmap")
    H = build_heatmap_hourly(d, metric, selected_channels)
    if not H.empty:
        fig_h = px.imshow(H, aspect="auto", labels=dict(x="Hour", y="Day", color=metric))
        st.plotly_chart(fig_h, use_container_width=True)

# =========================
# Channel
# =========================
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = df_long[
        (df_long["channel"] == ch) &
        (df_long["timestamp"] >= start_ts) &
        (df_long["timestamp"] <= end_ts)
    ].copy()
    if ch_df.empty:
        st.warning("No data for this channel in selected period."); st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df)
    cur  = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",  f"{pct_delta(cur['Sales'],  prev['Sales']):+.1f}%" if prev['Sales'] else None)
    C[1].metric("Orders",f"{cur['Orders']:,.0f}", f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None)
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}", f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None)
    C[3].metric("sale_ro (Sales/Ads)", "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None)
    C[4].metric("ads_ro (avg>0)", "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None)
    st.caption(f"Snapshot hour: {cur_hour}")

    st.markdown("### Trend overlay by day (channel)")
    met = st.selectbox("Metric to plot (channel):", ["sales","orders","ads","sale_ro","ads_ro"], index=0)
    piv = build_overlay_by_day_hour(ch_df, met, [ch])
    if not piv.empty:
        fig = go.Figure()
        for day in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[day], name=str(day), mode="lines+markers"))
        fig.update_layout(height=420, xaxis_title="Hour", yaxis_title=met)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Time series table (hourly)")
    H = build_heatmap_hourly(ch_df, met, [ch])
    if not H.empty:
        tbl = H.reset_index().melt(id_vars="index", var_name="hour", value_name=met).rename(columns={"index":"day"})
        st.dataframe(tbl, use_container_width=True, height=360)

# =========================
# Compare (คงโครงง่าย ๆ)
# =========================
else:
    picks = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:2], max_selections=4)
    if len(picks) < 2:
        st.info("Please pick at least 2 channels."); st.stop()

    sub = df_long[
        (df_long["channel"].isin(picks)) &
        (df_long["timestamp"] >= start_ts) &
        (df_long["timestamp"] <= end_ts)
    ].copy()
    if sub.empty:
        st.warning("No data for selected channels in range."); st.stop()

    st.subheader(f"Compare: {', '.join(picks)}")

    piv = build_overlay_by_day_hour(sub, "sale_ro", picks)
    if not piv.empty:
        base = st.selectbox("Baseline day", options=list(piv.columns), index=0)
        rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0
        fig = go.Figure()
        for c in rel.columns:
            if c == base: 
                continue
            fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
        fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference")
        st.plotly_chart(fig, use_container_width=True)
