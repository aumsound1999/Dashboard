# app.py
# Shopee ROAS Dashboard — overview • channel • compare
# Data source: Google Sheet CSV (Secrets: ROAS_CSV_URL = ".../gviz/tq?tqx=out:csv&sheet=...")

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers: detect time columns + parse
# -----------------------------------------------------------------------------
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g. "D21 12:45"


def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None


def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
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
    # รับค่าเป็นสตริงตัวเลขคั่นด้วย comma -> list ความยาว 6
    # เช่น "2025,12,34,776,22.51,1036"
    if not isinstance(s, str):
        return [np.nan] * 6
    if not re.search(r"\d", s):
        return [np.nan] * 6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]
    nums = []
    for p in parts[:6]:
        try:
            nums.append(float(p))
        except Exception:
            nums.append(np.nan)
    while len(nums) < 6:
        nums.append(np.nan)
    return nums


# -----------------------------------------------------------------------------
# Loaders (cached)
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
    df = pd.read_csv(io.StringIO(text))
    return df


def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # หา id + time cols
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    # melt
    df_melt = df_wide.melt(
        id_vars=id_cols, value_vars=time_cols,
        var_name="time_col", value_name="raw"
    )
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))

    # parse metrics -> v0..v5
    V = pd.DataFrame(df_melt["raw"].apply(parse_metrics_cell).tolist(),
                     columns=["v0", "v1", "v2", "v3", "v4", "v5"])

    out = pd.concat([df_melt[["timestamp"] + id_cols], V], axis=1).rename(columns={"name": "channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # mapping metrics
    out["sales"]  = pd.to_numeric(out["v0"], errors="coerce")    # 2025
    out["orders"] = pd.to_numeric(out["v1"], errors="coerce")    # 12
    out["ads"]    = pd.to_numeric(out["v2"], errors="coerce")    # 34 (budget/ads)
    out["view"]   = pd.to_numeric(out["v3"], errors="coerce")    # 776
    out["ads_ro"] = pd.to_numeric(out["v4"], errors="coerce")    # 22.51 (ads ro)
    # v5 misc ไม่ใช้

    # RO รวมแบบสะสม (ใช้เฉพาะที่จำเป็น)
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)

    return out


@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)


# -----------------------------------------------------------------------------
# Snapshot helpers
# -----------------------------------------------------------------------------
def latest_per_hour(df: pd.DataFrame, by_channel: bool = True) -> pd.DataFrame:
    """เลือกค่า snapshot ล่าสุดของแต่ละชั่วโมง (และต่อช่อง ถ้า by_channel=True)"""
    if df.empty:
        return df
    tmp = df.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    if by_channel:
        idx = tmp.sort_values("timestamp").groupby(["channel", "hour"]).tail(1).index
    else:
        idx = tmp.sort_values("timestamp").groupby(["hour"]).tail(1).index
    return tmp.loc[idx].copy()


def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    tz = str(df["timestamp"].dt.tz)
    target = at_ts.tz_convert(tz).floor("H")
    snap = (
        df[df["timestamp"].dt.floor("H") == target]
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
    return cur_snap, y_snap, cur_ts.floor("H")


def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales"].sum()
    orders = snap["orders"].sum()
    ads = snap["ads"].sum()
    sale_ro = (sales / ads) if ads != 0 else np.nan
    ads_ro_vals = snap["ads_ro"]
    ads_ro_avg = ads_ro_vals[ads_ro_vals > 0].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)


def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev):
        return None
    if curr is None or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev


# -----------------------------------------------------------------------------
# Metric transforms (core)
# -----------------------------------------------------------------------------
METRIC_LIST = ["sales", "orders", "ads", "sale_ro", "ads_ro"]


def hourly_delta_by_channel_day(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    แปลงค่า 'สะสม' เป็น 'เพิ่มขึ้นต่อชั่วโมง' ต่อวัน/ต่อช่อง
    - ถ้าติดลบจะ clip เป็น 0
    return columns: channel, day, h, delta
    """
    tmp = latest_per_hour(df, by_channel=True)
    tmp["day"] = tmp["hour"].dt.floor("D")
    tmp["h"] = tmp["hour"].dt.hour

    tmp = tmp.sort_values(["channel", "day", "hour"])
    # ค่าเดิมต่อช่อง/ต่อวัน
    tmp["prev"] = tmp.groupby(["channel", "day"])[value_col].shift(1)
    tmp["delta"] = tmp[value_col] - tmp["prev"]
    tmp.loc[(tmp["delta"].isna()) | (tmp["delta"] < 0), "delta"] = 0.0

    return tmp[["channel", "day", "h", "delta"]]


def ro_hourly(df: pd.DataFrame, ro_name: str = "sale_ro") -> pd.DataFrame:
    """
    RO รายชั่วโมงจริง (weighted):
    - sale_ro: sum(delta_sales) / sum(delta_ads)
    - ads_ro:  sum(delta_sales_from_ads?) -> ใช้สูตรเดียวกัน (ยึด delta_sales/delta_ads)
    clip ที่ 50
    return columns: day, h, ro
    """
    d_sales = hourly_delta_by_channel_day(df, "sales")
    d_ads   = hourly_delta_by_channel_day(df, "ads")

    # รวมรายชั่วโมงทุกช่อง
    s = d_sales.groupby(["day", "h"])["delta"].sum().rename("d_sales")
    a = d_ads.groupby(["day", "h"])["delta"].sum().rename("d_ads")
    ro = pd.concat([s, a], axis=1).reset_index()
    ro["ro"] = ro["d_sales"] / ro["d_ads"].replace(0, np.nan)
    ro.loc[ro["ro"] > 50, "ro"] = 50.0
    ro["ro"] = ro["ro"].fillna(0.0)

    return ro[["day", "h", "ro"]]


def overlay_series_by_day(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    สร้างซีรีส์สำหรับ 'Trend overlay by day' ตาม metric ที่เลือก:
    - sales/orders/ads -> ใช้ delta รวมทุกช่องต่อชั่วโมง
    - sale_ro/ads_ro   -> ro_hourly (weighted) clip 50
    return columns: day, h, val
    """
    if metric in ["sales", "orders", "ads"]:
        deltas = hourly_delta_by_channel_day(df, metric)
        # รวมทุกช่องต่อชั่วโมง
        agg = deltas.groupby(["day", "h"])["delta"].sum().reset_index().rename(columns={"delta": "val"})
        return agg

    elif metric in ["sale_ro", "ads_ro"]:
        ro = ro_hourly(df, metric)
        return ro.rename(columns={"ro": "val"})

    else:
        raise ValueError(f"Unknown metric: {metric}")


def heatmap_pivot_from_overlay(overlay_df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ overlay_df (day,h,val) -> Pivot เป็น Day x Hour
    """
    if overlay_df.empty:
        return pd.DataFrame()
    pvt = overlay_df.pivot(index="day", columns="h", values="val").sort_index(ascending=False)
    # ช่องที่หาย/ค่าว่าง -> 0
    pvt = pvt.fillna(0.0)
    # เรียงคอลัมน์ให้ครบ 0..23
    all_hours = list(range(24))
    for h in all_hours:
        if h not in pvt.columns:
            pvt[h] = 0.0
    pvt = pvt[all_hours]
    return pvt


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Reload cache
if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear()
    load_wide_df.clear()
    build_long.clear()
    st.experimental_rerun()

# Load data
try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.stop()

tz = "Asia/Bangkok"
now_ts = pd.Timestamp.now(tz=tz)

# Date range default 3 days
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
if isinstance(d1, (list, tuple)):  # streamlit quirk
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

# Channels [All]
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# Filtered working set
mask = (
    (df_long["timestamp"] >= start_ts)
    & (df_long["timestamp"] <= end_ts)
    & (df_long["channel"].isin(selected_channels))
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

    # KPI snapshot (current vs yesterday)
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

    # --------- Trend overlay by day (ตาม metric เดียวกับ heatmap) ---------
    st.markdown("### Trend overlay by day")
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", METRIC_LIST, index=0, key="ov_metric")

    overlay = overlay_series_by_day(d, metric)
    if overlay.empty:
        st.info("No data to plot.")
    else:
        # สร้างเส้นซ้อนกันตามวัน
        fig = go.Figure()
        for day, g in overlay.groupby("day"):
            g = g.sort_values("h")
            x = [f"{int(h):02d}:00" for h in g["h"]]
            fig.add_trace(go.Scatter(x=x, y=g["val"], mode="lines+markers", name=str(day.date())))
        fig.update_layout(
            height=420,
            xaxis_title="Time (HH:MM)",
            yaxis_title=metric,
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------- Prime hours heatmap (ผูก metric เดียวกัน) ---------
    st.markdown("### Prime hours heatmap")
    pvt = heatmap_pivot_from_overlay(overlay)
    if pvt.empty:
        st.info("No data for heatmap.")
    else:
        # color range: ro 0..50, อื่น ๆ ใช้เปอร์เซ็นไทล์ลด outlier
        color_range = None
        color_label = metric
        if metric in ["sale_ro", "ads_ro"]:
            zmin, zmax = 0, 50
        else:
            low = np.nanpercentile(pvt.values, 5)
            high = np.nanpercentile(pvt.values, 95)
            zmin, zmax = float(low), float(high)
            if zmax <= zmin:
                zmin, zmax = 0.0, float(pvt.values.max())

        fig_h = px.imshow(
            pvt,
            aspect="auto",
            color_continuous_scale="Blues",
            zmin=zmin, zmax=zmax,
            labels=dict(x="Hour", y="Day", color=color_label),
        )
        # สวยงามเพิ่ม
        fig_h.update_layout(height=450)
        st.plotly_chart(fig_h, use_container_width=True)

    # --------- (Optional) Raw table: เคย error — เว้นไว้ก่อนตามที่คุย ---------
    # st.markdown("#### Data (hourly latest snapshot per channel)")
    # show = latest_per_hour(d, by_channel=True)[["hour", "channel", "ads", "orders", "sales", "SaleRO", "ads_ro"]]
    # st.dataframe(show.sort_values(["hour", "channel"]).round(3), use_container_width=True, height=360)

# -----------------------------------------------------------------------------
# Channel
# -----------------------------------------------------------------------------
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = df_long[
        (df_long["channel"] == ch)
        & (df_long["timestamp"] >= start_ts)
        & (df_long["timestamp"] <= end_ts)
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

    # กราฟซ้อนวัน (สำหรับช่องเดียว) — ใช้ metric dropdown เดียวกัน
    st.markdown("### Trend overlay by day (channel)")
    metric_ch = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", METRIC_LIST, index=0, key="ch_metric")
    overlay_ch = overlay_series_by_day(ch_df, metric_ch)
    if overlay_ch.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for day, g in overlay_ch.groupby("day"):
            g = g.sort_values("h")
            x = [f"{int(h):02d}:00" for h in g["h"]]
            fig.add_trace(go.Scatter(x=x, y=g["val"], mode="lines+markers", name=str(day.date())))
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=metric_ch, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    # ตาราง Time-series (snapshot/ชั่วโมง) ของช่องเดียว — คงไว้เหมือนเดิม
    st.markdown("### Time series table")
    ch_hourly = latest_per_hour(ch_df, by_channel=False).sort_values("hour")
    ch_hourly["sale_ro"] = ch_hourly["sales"] / ch_hourly["ads"].replace(0, np.nan)
    show = ch_hourly[["hour", "ads", "orders", "sales", "sale_ro", "ads_ro"]].rename(columns={"ads": "budget(ads)"})
    st.dataframe(show.round(3), use_container_width=True, height=360)

# -----------------------------------------------------------------------------
# Compare (คงหลักการเดิม)
# -----------------------------------------------------------------------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:2], max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()

    sub = df_long[
        (df_long["channel"].isin(pick))
        & (df_long["timestamp"] >= start_ts)
        & (df_long["timestamp"] <= end_ts)
    ].copy()
    if sub.empty:
        st.warning("No data for selected channels in range.")
        st.stop()

    tmp = latest_per_hour(sub, by_channel=True)

    st.subheader(f"Compare: {', '.join(pick)}")
    # KPI table แบบง่าย
    kpis = tmp.groupby("channel").agg(
        ROAS=("SaleRO", "mean"),
        AOV=("sales", lambda s: (s.sum() / tmp.loc[s.index, "orders"].sum()) if tmp.loc[s.index, "orders"].sum() else np.nan),
        CPO=("orders", lambda s: (tmp.loc[s.index, "ads"].sum() / s.sum()) if s.sum() else np.nan),
        RPV=("sales", lambda s: (s.sum() / tmp.loc[s.index, "view"].sum()) if tmp.loc[s.index, "view"].sum() else np.nan),
        ORV=("orders", lambda s: (s.sum() / tmp.loc[s.index, "view"].sum()) if tmp.loc[s.index, "view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS", "sales", "orders", "ads"], index=0)

    piv = tmp.pivot_table(index="hour", columns="channel", values=("SaleRO" if met == "ROAS" else met), aggfunc="sum").sort_index()
    rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0

    fig = go.Figure()
    for c in rel.columns:
        if c == base:
            continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference", legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = tmp.pivot_table(index="hour", columns="channel", values="SaleRO", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
