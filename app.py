# app.py
# Shopee ROAS Dashboard (Overview • Channel • Compare)
# ต้องมี Secrets: ROAS_CSV_URL = "...gviz/tq?tqx=out:csv&sheet=..."
import os, re, io
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers: parse & load
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
    """string '2025,12,34,776,22.51,1036' -> 6 ตัวเลข (padding NaN ถ้าไม่ครบ)"""
    if not isinstance(s, str) or not re.search(r"\d", s):
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
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    m = df_wide.melt(id_vars=id_cols, value_vars=time_cols, var_name="time_col", value_name="raw")
    m["timestamp"] = m["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz))
    parsed = m["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([m[["timestamp"] + id_cols], V], axis=1).rename(columns={"name":"channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # mapping:
    out["sales"]  = pd.to_numeric(out["v0"], errors="coerce")  # 2025
    out["orders"] = pd.to_numeric(out["v1"], errors="coerce")  # 12
    out["ads"]    = pd.to_numeric(out["v2"], errors="coerce")  # 34
    out["view"]   = pd.to_numeric(out["v3"], errors="coerce")  # 776
    out["ads_ro_raw"] = pd.to_numeric(out["v4"], errors="coerce")  # 22.51

    # sale_ro ต่อแถว
    out["sale_ro_raw"] = out["sales"] / out["ads"].replace(0, np.nan)
    out["ads_ro"]  = out["ads_ro_raw"]
    out["SaleRO"]  = out["sale_ro_raw"]
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------
def cap_ro(x, cap=50.0):
    return np.minimum(x, cap)

def pick_latest_per_hour(df: pd.DataFrame, by_cols):
    """คืนข้อมูล 'สแน็ปล่าสุดต่อชั่วโมง' ตาม keys ใน by_cols (เช่น ['channel','day','h'])"""
    if df.empty:
        return df
    tmp = df.copy()
    tmp["hour_floor"] = tmp["timestamp"].dt.floor("H")
    # groupby keys + hour แล้วเอาแถวล่าสุด
    idx = tmp.sort_values("timestamp").groupby(by_cols + ["hour_floor"]).tail(1).index
    return tmp.loc[idx].copy()

def overlay_series_by_day(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    คืนตาราง  day, h, val  สำหรับวาดเส้นซ้อนระหว่างวัน
    - sales/orders/ads: sum per ('day','h')
    - sale_ro: mean ของ sale_ro ต่อช่อง (cap 50)
    - ads_ro : mean ของ ads_ro ต่อช่อง (cap 50) (เฉพาะค่า > 0)
    """
    if df.empty:
        return pd.DataFrame(columns=["day","h","val"])

    d = df.copy()
    d["day"] = d["timestamp"].dt.date
    d["h"]   = d["timestamp"].dt.hour

    latest = pick_latest_per_hour(d, ["channel","day","h"])

    if metric in ["sales", "orders", "ads"]:
        g = latest.groupby(["day","h"], as_index=False)[metric].sum()
        g = g.rename(columns={metric:"val"})
    elif metric == "sale_ro":
        latest["ro"] = cap_ro(latest["SaleRO"])
        g = (latest.groupby(["day","h"])["ro"]
                   .mean()
                   .reset_index()
                   .rename(columns={"ro":"val"}))
    elif metric == "ads_ro":
        latest["ro"] = cap_ro(latest["ads_ro"])
        g = (latest.loc[latest["ro"]>0]
                   .groupby(["day","h"])["ro"]
                   .mean()
                   .reset_index()
                   .rename(columns={"ro":"val"}))
    else:
        return pd.DataFrame(columns=["day","h","val"])

    return g.sort_values(["day","h"])

def build_heatmap_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    ตาราง pivot สำหรับ heatmap (index=day, columns=h)
    - sales/orders/ads: ใช้ 'ชั่วโมงนี้ - ชั่วโมงก่อน' บนยอดรวมทุกช่องต่อวัน
    - sale_ro / ads_ro: ใช้ค่ารายชั่วโมง (cap 50) (ads_ro ใช้เฉพาะค่า > 0)
    """
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["day"] = d["timestamp"].dt.date
    d["h"]   = d["timestamp"].dt.hour

    # ล่าสุดต่อช่อง/ชั่วโมง
    latest = pick_latest_per_hour(d, ["channel","day","h"])

    if metric in ["sales","orders","ads"]:
        agg = latest.groupby(["day","h"], as_index=False)[metric].sum()
        # คำนวณ hour-over-hour ต่อวัน
        agg["val"] = agg.groupby("day")[metric].diff()
        # ถ้าชั่วโมงแรกของวัน จะเป็น NaN -> ใส่ 0 (หรือปล่อย NaN ก็ได้)
        agg["val"] = agg["val"].fillna(0)
        M = agg.pivot(index="day", columns="h", values="val")
    elif metric == "sale_ro":
        latest["ro"] = cap_ro(latest["SaleRO"])
        agg = latest.groupby(["day","h"], as_index=False)["ro"].mean()
        M = agg.pivot(index="day", columns="h", values="ro")
    elif metric == "ads_ro":
        latest["ro"] = cap_ro(latest["ads_ro"])
        agg = latest.loc[latest["ro"]>0].groupby(["day","h"], as_index=False)["ro"].mean()
        M = agg.pivot(index="day", columns="h", values="ro")
    else:
        return pd.DataFrame()

    # เรียงชั่วโมง 0..23
    M = M.reindex(columns=list(range(24)))
    return M

# -----------------------------------------------------------------------------
# Plotting helper: เส้นซ้อนกันตามวัน (แก้ปัญหาเส้นมั่ว)
# -----------------------------------------------------------------------------
def plot_overlay_by_day(overlay_df: pd.DataFrame, metric_label: str, height=420):
    """
    overlay_df: columns => day, h, val
    - แกน X เป็นตัวเลข 0..23
    - เติม None สำหรับชั่วโมงที่ขาด (ไม่เชื่อมข้าม)
    """
    hours = list(range(24))
    ticktext = [f"{h:02d}:00" for h in hours]

    fig = go.Figure()
    for day, g in overlay_df.groupby("day"):
        arr = [None] * 24
        for _, r in g.iterrows():
            hh = int(r["h"])
            if 0 <= hh <= 23:
                arr[hh] = r["val"]
        fig.add_trace(
            go.Scatter(
                x=hours, y=arr,
                mode="lines+markers",
                name=str(pd.to_datetime(day).date()),
                connectgaps=False
            )
        )

    fig.update_layout(
        height=height,
        xaxis=dict(tickmode="array", tickvals=hours, ticktext=ticktext, range=[-0.5, 23.5]),
        yaxis_title=metric_label,
        xaxis_title="Time (HH:MM)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=10, b=40),
    )
    return fig

# -----------------------------------------------------------------------------
# KPI helpers
# -----------------------------------------------------------------------------
def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    tz = str(df["timestamp"].dt.tz)
    target_hour = at_ts.tz_convert(tz).floor("H")
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
    sale_ro = (sales / ads) if ads else np.nan
    ads_ro_vals = snap["ads_ro_raw"]
    ads_ro_avg = ads_ro_vals[ads_ro_vals > 0].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev):
        return None
    if curr is None or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

# -----------------------------------------------------------------------------
# Load data + sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear(); load_wide_df.clear(); build_long.clear()
    st.experimental_rerun()

try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
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
    min_value=min_ts.date(), max_value=date_max,
)
if isinstance(d1, (list, tuple)):
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

mask = (
    (df_long["timestamp"] >= start_ts)
    & (df_long["timestamp"] <= end_ts)
    & (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

metric_opts = {
    "sales": "sales",
    "orders": "orders",
    "Budget (ads)": "ads",
    "sale_ro": "sale_ro",
    "ads_ro": "ads_ro",
}

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

    st.markdown("### Trend overlay by day")
    metric_key = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", list(metric_opts.keys()), index=0)
    metric = metric_opts[metric_key]

    overlay = overlay_series_by_day(d, metric)
    if overlay.empty:
        st.info("No data to plot.")
    else:
        fig = plot_overlay_by_day(overlay, metric_label=metric, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prime hours heatmap")
    M = build_heatmap_matrix(d, metric)
    if M.empty:
        st.info("No data for heatmap.")
    else:
        fig_h = px.imshow(
            M.sort_index(ascending=False),
            aspect="auto",
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

    st.markdown("### Trend overlay by day (channel)")
    metric_key_ch = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", list(metric_opts.keys()), index=0, key="metric_ch")
    metric_ch = metric_opts[metric_key_ch]

    overlay_ch = overlay_series_by_day(ch_df, metric_ch)
    if overlay_ch.empty:
        st.info("No data to plot.")
    else:
        fig = plot_overlay_by_day(overlay_ch, metric_label=metric_ch, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Time series heatmap (channel)")
    Mch = build_heatmap_matrix(ch_df, metric_ch)
    if Mch.empty:
        st.info("No data for heatmap.")
    else:
        fig_h = px.imshow(
            Mch.sort_index(ascending=False),
            aspect="auto",
            labels=dict(x="Hour", y="Day", color=metric_ch),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_h, use_container_width=True)

# -----------------------------------------------------------------------------
# Compare (คงไว้แบบย่อ)
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

    # สแนปฯ ต่อชั่วโมง/ช่อง
    sub["hour"] = sub["timestamp"].dt.floor("H")
    idx = sub.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    hourly = sub.loc[idx]

    st.subheader(f"Compare: {', '.join(pick)}")
    kpis = hourly.groupby("channel").agg(
        ROAS=("SaleRO", lambda s: np.nanmean(cap_ro(s))),
        SALES=("sales","sum"),
        ORDERS=("orders","sum"),
        ADS=("ads","sum"),
    ).reset_index()
    st.dataframe(kpis.round(3), use_container_width=True)
