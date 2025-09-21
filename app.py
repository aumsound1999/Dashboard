# app.py
# Shopee ROAS Dashboard (Overview / Channel / Compare)
# - Fix line-plot ordering ("เส้นมั่ว") by using last-snapshot-per-hour, day-grouped sorting
# - Compute hourly sale_ro and ads_ro from deltas:
#       sale_ro_hour = ΔSales / ΔAds
#       ads_ro_hour  = Δ(ROAS * Ads) / ΔAds
#   and clip at 50 for charts/heatmap
#
# Requirements (requirements.txt):
#   streamlit
#   pandas
#   numpy
#   plotly
#   requests

import os, io, re
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

TZ = "Asia/Bangkok"
st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers: header parsing for time columns
# -----------------------------------------------------------------------------
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = TZ) -> pd.Timestamp:
    # "D21 12:4" -> day=21, hour=12, minute=4
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
    """
    รับ string ตัวเลขคั่นด้วย comma -> list(6)
    เช่น: "2025,12,34,776,22.51,1036" -> [2025,12,34,776,22.51,1036]
    """
    if not isinstance(s, str) or not re.search(r"\d", s):
        return [np.nan] * 6
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
    return nums

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Transform: wide -> long + engineering
# -----------------------------------------------------------------------------
def wide_to_long(df_wide: pd.DataFrame, tz: str = TZ) -> pd.DataFrame:
    # 1) identify columns
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)

    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    m = df_wide.melt(
        id_vars=id_cols, value_vars=time_cols,
        var_name="time_col", value_name="raw"
    )
    m["timestamp"] = m["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))
    parsed = m["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])
    out = pd.concat([m[["timestamp"] + id_cols], V], axis=1)
    out = out.rename(columns={"name": "channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Mapping ตามชีตของคุณ:
    # v0=sales (สะสม), v1=orders(สะสม), v2=ads spend(สะสม), v3=view(สะสม), v4=ads_ro (ROAS สะสม/เฉลี่ย)
    out["sales_cum"]  = pd.to_numeric(out["v0"], errors="coerce")
    out["orders_cum"] = pd.to_numeric(out["v1"], errors="coerce")
    out["ads_cum"]    = pd.to_numeric(out["v2"], errors="coerce")
    out["view_cum"]   = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro_avg"] = pd.to_numeric(out["v4"], errors="coerce")  # ROAS avg until now

    # ทำคอลัมน์เวลาใช้งานสะดวก
    out["ts_local"] = out["timestamp"].dt.tz_convert(tz)
    out["hour_local"] = out["ts_local"].dt.floor("H")           # ชม. (ใช้กราฟ/heatmap)
    out["day"] = out["ts_local"].dt.date                        # วัน (group เพื่อกันข้ามวัน)

    # เก็บ "ล่าสุดต่อชั่วโมง/ช่อง" เพื่อกันกราฟเส้นกระโดด
    out = out.sort_values(["channel", "timestamp"])
    idx = out.groupby(["channel", "hour_local"]).tail(1).index
    out = out.loc[idx].reset_index(drop=True)

    # คำนวณยอดขายจากแอดสะสม: R_t = ROAS_t * Ads_t
    out["ads_rev_cum"] = out["ads_ro_avg"] * out["ads_cum"]

    # หา delta รายชั่วโมงต่อวัน/ช่อง
    out = out.sort_values(["channel","day","hour_local"])
    out["d_sales"]     = out.groupby(["channel","day"])["sales_cum"].diff()
    out["d_orders"]    = out.groupby(["channel","day"])["orders_cum"].diff()
    out["d_ads"]       = out.groupby(["channel","day"])["ads_cum"].diff()
    out["d_ads_rev"]   = out.groupby(["channel","day"])["ads_rev_cum"].diff()

    # ROAS รายชั่วโมง (สูตรใหม่)
    out["sale_ro_hour"] = np.where(out["d_ads"]>0, out["d_sales"] / out["d_ads"], np.nan)
    out["ads_ro_hour"]  = np.where(out["d_ads"]>0, out["d_ads_rev"] / out["d_ads"], np.nan)

    # เวอร์ชัน clip สำหรับกราฟ/heatmap
    out["sale_ro_hour_clip"] = out["sale_ro_hour"].clip(upper=50)
    out["ads_ro_hour_clip"]  = out["ads_ro_hour"].clip(upper=50)

    return out

@st.cache_data(ttl=600)
def build_long(df_wide):
    return wide_to_long(df_wide)

# -----------------------------------------------------------------------------
# KPI snapshots
# -----------------------------------------------------------------------------
def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    tz = str(df["ts_local"].dt.tz)
    target_hour = at_ts.tz_convert(tz).floor("H")
    snap = (
        df[df["hour_local"] == target_hour]
        .sort_values("timestamp")
        .groupby("channel")
        .tail(1)
    )
    return snap

def current_and_yesterday_snapshots(df: pd.DataFrame):
    if df.empty:
        return df, df, pd.NaT
    cur_ts = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts)
    y_snap = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1))
    return cur_snap, y_snap, cur_ts.tz_convert(TZ).floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales_cum"].sum()
    orders = snap["orders_cum"].sum()
    ads = snap["ads_cum"].sum()
    sale_ro = (sales / ads) if ads else np.nan
    ads_ro_avg = snap["ads_ro_avg"].replace(0, np.nan)
    ads_ro_avg = ads_ro_avg[ads_ro_avg>0].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev):
        return None
    if curr is None or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

# -----------------------------------------------------------------------------
# Builders for overlay/heatmap
# -----------------------------------------------------------------------------
def overlay_by_day(df: pd.DataFrame, metric: str, channels: list) -> pd.DataFrame:
    """
    เตรียมข้อมูลสำหรับกราฟซ้อนวัน (หนึ่ง metric)
    - สำหรับ sales/orders/ads -> ใช้ delta (ต่อชั่วโมง)
    - สำหรับ sale_ro/ads_ro    -> ratio-of-sums จาก deltas (ต่อชั่วโมง)
    """
    sub = df[df["channel"].isin(channels)].copy()
    if sub.empty:
        return sub

    # รวบรายชั่วโมง-รายวันข้ามหลายช่อง
    g = sub.groupby(["day","hour_local"])

    if metric in ["sales","orders","ads"]:
        col = {"sales":"d_sales", "orders":"d_orders","ads":"d_ads"}[metric]
        agg = g[col].sum().reset_index()
        agg["val"] = agg[col]
    elif metric == "sale_ro":
        X = g["d_sales"].sum()
        Y = g["d_ads"].sum().replace(0, np.nan)
        agg = pd.concat([X, Y], axis=1).reset_index().rename(columns={"d_sales":"num","d_ads":"den"})
        agg["val"] = (agg["num"] / agg["den"]).clip(upper=50)
    else:  # ads_ro
        X = g["d_ads_rev"].sum()
        Y = g["d_ads"].sum().replace(0, np.nan)
        agg = pd.concat([X, Y], axis=1).reset_index().rename(columns={"d_ads_rev":"num","d_ads":"den"})
        agg["val"] = (agg["num"] / agg["den"]).clip(upper=50)

    agg["time_str"] = agg["hour_local"].dt.strftime("%H:%M")
    return agg.sort_values(["day","hour_local"]).reset_index(drop=True)

def heatmap_day_hour(df: pd.DataFrame, metric: str, channels: list) -> pd.DataFrame:
    """
    ทำตาราง Day x Hour สำหรับ heatmap
    - sales/orders/ads -> ใช้ delta
    - sale_ro/ads_ro   -> ใช้ hourly ROAS (ratio-of-sums) และ clip=50
    """
    sub = df[df["channel"].isin(channels)].copy()
    if sub.empty:
        return sub

    g = sub.groupby(["day","hour_local"])
    if metric in ["sales","orders","ads"]:
        col = {"sales":"d_sales", "orders":"d_orders", "ads":"d_ads"}[metric]
        H = g[col].sum().reset_index()
        H["val"] = H[col]
    elif metric == "sale_ro":
        X = g["d_sales"].sum()
        Y = g["d_ads"].sum().replace(0, np.nan)
        H = pd.concat([X,Y], axis=1).reset_index().rename(columns={"d_sales":"num","d_ads":"den"})
        H["val"] = (H["num"]/H["den"]).clip(upper=50)
    else:
        X = g["d_ads_rev"].sum()
        Y = g["d_ads"].sum().replace(0, np.nan)
        H = pd.concat([X,Y], axis=1).reset_index().rename(columns={"d_ads_rev":"num","d_ads":"den"})
        H["val"] = (H["num"]/H["den"]).clip(upper=50)

    H["H"] = H["hour_local"].dt.hour
    return H.sort_values(["day","H"]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")
if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear()
    load_wide_df.clear()
    build_long.clear()
    st.experimental_rerun()

try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.stop()

now_ts = pd.Timestamp.now(tz=TZ)
min_ts = df_long["ts_local"].min()
max_ts = df_long["ts_local"].max()

# Date range (default 3 days)
date_max = max_ts.date()
date_min_default = (max_ts - pd.Timedelta(days=2)).date()
d1, d2 = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(date_min_default, date_max),
    min_value=min_ts.date(),
    max_value=date_max,
)
if isinstance(d1, (list, tuple)):
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(TZ)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(TZ)

# Channel filter ([All])
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
selected_channels = all_channels if ("[All]" in chosen or not any(c in all_channels for c in chosen)) \
                    else [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# Filter dataframe
mask = (
    (df_long["ts_local"] >= start_ts) &
    (df_long["ts_local"] <= end_ts) &
    (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d)
    cur = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",
                delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}",
                delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    C[3].metric("sale_ro (Sales/Ads)",
                "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    C[4].metric("ads_ro (avg>0)",
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # Trend overlay by day (single metric)
    st.markdown("### Trend overlay by day")
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                          options=["sales","orders","ads","sale_ro","ads_ro"],
                          index=0, key="ov_metric")
    ov = overlay_by_day(d, metric, selected_channels)
    if ov.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for dy, sub in ov.groupby("day"):
            fig.add_trace(
                go.Scatter(
                    x=sub["time_str"], y=sub["val"],
                    mode="lines+markers", name=str(dy)
                )
            )
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    # Prime hours heatmap (ตาม metric ที่เลือก)
    st.markdown("### Prime hours heatmap")
    hm = heatmap_day_hour(d, metric, selected_channels)
    if hm.empty:
        st.info("No data for heatmap.")
    else:
        pivot = hm.pivot(index="day", columns="H", values="val").sort_index(ascending=False)
        fig_h = px.imshow(
            pivot, aspect="auto",
            labels=dict(x="Hour", y="Day", color=metric),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_h, use_container_width=True)

# -----------------------------------------------------------------------------
# Channel
# -----------------------------------------------------------------------------
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = df_long[
        (df_long["channel"] == ch) &
        (df_long["ts_local"] >= start_ts) &
        (df_long["ts_local"] <= end_ts)
    ].copy()
    if ch_df.empty:
        st.warning("No data for this channel in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df)
    cur = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",
                delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}",
                delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    C[3].metric("sale_ro (Sales/Ads)",
                "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    C[4].metric("ads_ro (avg>0)",
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    st.markdown("### Trend overlay by day (channel)")
    met_ch = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                          options=["sales","orders","ads","sale_ro","ads_ro"],
                          index=0, key="ch_metric")
    ov_ch = overlay_by_day(ch_df, met_ch, [ch])
    if ov_ch.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for dy, sub in ov_ch.groupby("day"):
            fig.add_trace(go.Scatter(x=sub["time_str"], y=sub["val"], mode="lines+markers", name=str(dy)))
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=met_ch)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Time series table")
    # แสดง delta/roas รายชั่วโมงของช่องเดียว (เพื่อดีบัก/ดูดิบ)
    t = ch_df.sort_values(["day","hour_local"])[
        ["day","hour_local","d_sales","d_orders","d_ads","sale_ro_hour","ads_ro_hour"]
    ].copy()
    t["hour"] = t["hour_local"].dt.strftime("%Y-%m-%d %H:%M")
    t = t.drop(columns=["hour_local"]).rename(
        columns={"d_sales":"sales","d_orders":"orders","d_ads":"ads",
                 "sale_ro_hour":"sale_ro","ads_ro_hour":"ads_ro"})
    st.dataframe(t.round(3), use_container_width=True, height=360)

# -----------------------------------------------------------------------------
# Compare (simple relative line)
# -----------------------------------------------------------------------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:2], max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()

    sub = df_long[
        (df_long["channel"].isin(pick)) &
        (df_long["ts_local"] >= start_ts) &
        (df_long["ts_local"] <= end_ts)
    ].copy()
    if sub.empty:
        st.warning("No data for selected channels.")
        st.stop()

    # สร้าง ROAS รายชั่วโมง/เดลต้าตามสูตรใหม่แล้วใน df_long อยู่แล้ว
    st.subheader(f"Compare: {', '.join(pick)}")
    met = st.selectbox("Metric", options=["sale_ro","ads_ro","sales","orders","ads"], index=0)

    ov_cmp = overlay_by_day(sub, met, pick)
    if ov_cmp.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for dy, subd in ov_cmp.groupby("day"):
            fig.add_trace(go.Scatter(x=subd["time_str"], y=subd["val"], mode="lines", name=str(dy)))
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=met)
        st.plotly_chart(fig, use_container_width=True)
