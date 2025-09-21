# app.py
# Shopee ROAS Dashboard — overview • channel • compare
# Data is read from a Google Sheet (CSV export) provided via Secrets:
#   ROAS_CSV_URL = "https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>"
#
# Install (requirements.txt):
#   streamlit
#   pandas
#   numpy
#   plotly
#   requests

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import timedelta

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# ============================================================================#
# Helpers: load + parse
# ============================================================================#

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
    """'D21 12:4' -> day=21, hour=12, minute=4 ; year/month = today"""
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
    รับค่าเป็นสตริงตัวเลขคั่นด้วยคอมมา -> list ความยาว 6 (เติม NaN ถ้าไม่ครบ)
    เช่น '2025,12,34,776,22.51,1036'
    """
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
    # หา id cols + time cols
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="raw")
    # timestamp
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))

    # metrics -> v0..v5
    parsed = df_melt["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([df_melt[["timestamp"]+id_cols], V], axis=1).rename(columns={"name":"channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # ---- mapping (แก้ตรงนี้ถ้าลำดับในชีตต่างไป) ----
    out["sales"]  = pd.to_numeric(out["v0"], errors="coerce")
    out["orders"] = pd.to_numeric(out["v1"], errors="coerce")
    out["ads"]    = pd.to_numeric(out["v2"], errors="coerce")  # ค่าโฆษณา/งบ
    out["view"]   = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro"] = pd.to_numeric(out["v4"], errors="coerce")  # RO ของ ads
    out["misc"]   = pd.to_numeric(out["v5"], errors="coerce")

    # ROAS รวม: sale_ro = sales/ads
    out["sale_ro"] = out["sales"] / out["ads"].replace(0, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    """ล่าสุดของแต่ละ channel ภายในชั่วโมงเป้าหมาย"""
    if df.empty:
        return df
    tz = str(df["timestamp"].dt.tz)
    target_hour = at_ts.tz_convert(tz).floor("H")
    snap = (df[df["timestamp"].dt.floor("H") == target_hour]
            .sort_values("timestamp")
            .groupby("channel")
            .tail(1))
    return snap

def current_and_yesterday_snapshots(df: pd.DataFrame):
    if df.empty:
        return df, df, pd.NaT
    cur_ts = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts)
    y_snap = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1))
    return cur_snap, y_snap, cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    """รวมค่าแบบใหม่สำหรับหัว KPI"""
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
    if prev in [0, None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr): return None
    return (curr - prev) * 100.0 / prev

# ---------- utility: สร้าง hourly snapshot และตาราง overlay ต่อวัน ----------
def build_hourly_snapshot(df: pd.DataFrame, by_channel=True):
    """
    คืนค่า snapshot ต่อชั่วโมง:
    - ถ้า by_channel=True: snapshot ต่อ channel -> ใช้ต่อสำหรับ aggregate ข้ามช่อง
    - ถ้า False: (เช่น channel page) จะ snapshot ต่อชั่วโมงตรง ๆ
    """
    tmp = df.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    if by_channel:
        idx = tmp.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    else:
        idx = tmp.sort_values("timestamp").groupby("hour").tail(1).index
    hourly = tmp.loc[idx].copy()
    return hourly

def make_overlay_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    สร้างตาราง overlay ต่อวันต่อเวลา (HH:MM)
    - metrics ที่รวมแบบ sum: sales, orders, ads
    - ads_ro: เฉลี่ยเฉพาะค่าที่ > 0
    - sale_ro: คำนวณจากยอดรวม sales/ads ของ slot นั้น
    """
    df = hourly.copy()
    df["day"]  = df["hour"].dt.date
    df["hhmm"] = df["hour"].dt.strftime("%H:%M")
    g = df.groupby(["day","hhmm"]).agg(
        sales = ("sales","sum"),
        orders= ("orders","sum"),
        ads   = ("ads","sum"),
        ads_ro=("ads_ro", lambda s: s[s>0].mean())
    ).reset_index()
    g["sale_ro"] = g["sales"] / g["ads"].replace(0, np.nan)
    return g

# ============================================================================#
# UI: Data refresh header (Reload only)
# ============================================================================#

st.sidebar.header("Filters")

# ปุ่มรีโหลด (ล้าง cache ทั้งหมด)
col_top = st.columns([1, 3, 3])[0]
with col_top:
    if st.button("Reload", use_container_width=True):
        fetch_csv_text.clear()
        load_wide_df.clear()
        build_long.clear()
        st.experimental_rerun()

# โหลดข้อมูล
try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.dataframe(wide.head() if 'wide' in locals() else pd.DataFrame())
    st.stop()

tz = "Asia/Bangkok"
now_ts = pd.Timestamp.now(tz=tz)

# ============================================================================#
# Sidebar filters — default 3 days, Channels [All]
# ============================================================================#

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
if isinstance(d1, (list, tuple)):
    d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

# Channels with [All]
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# กรองช่วงวันที่ + ช่อง
mask = (
    (df_long["timestamp"] >= start_ts)
    & (df_long["timestamp"] <= end_ts)
    & (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

# แสดงหัวเรื่อง + last refresh
st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================#
# Overview
# ============================================================================#
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period.")
        st.stop()

    # สแนปช็อตล่าสุด และเมื่อวานเวลาเดียวกัน
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
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}%",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # ---- hourly snapshot (ต่อ channel) เพื่อรวมข้ามช่องได้ถูกต้อง ----
    hourly_ch = build_hourly_snapshot(d, by_channel=True)

    # ===== OVERLAY (day-by-day, single metric) =====
    st.markdown("#### Trend overlay by day")
    metric_label = st.selectbox(
        "Metric to plot (เลือกได้ 1 ค่า)",
        options=["Sales","Orders","Budget (Ads)","sale_ro","ads_ro"],
        index=0
    )
    col_map = {
        "Sales": "sales",
        "Orders": "orders",
        "Budget (Ads)": "ads",
        "sale_ro": "sale_ro",
        "ads_ro": "ads_ro",
    }

    overlay = make_overlay_table(hourly_ch)  # รวมข้ามช่องแล้ว (ต่อวัน-ต่อเวลา)
    ycol = col_map[metric_label]

    fig_overlay = go.Figure()
    for day in sorted(overlay["day"].unique()):
        sub = overlay[overlay["day"] == day].sort_values("hhmm")
        fig_overlay.add_trace(go.Scatter(
            x=sub["hhmm"], y=sub[ycol],
            name=str(day), mode="lines+markers"
        ))
    fig_overlay.update_layout(
        height=420,
        xaxis_title="Time (HH:MM)",
        yaxis_title=metric_label,
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    # ===== ตาราง (hourly latest snapshot per channel) =====
    st.markdown("#### Data (hourly latest snapshot per channel)")
    # รวมสรุปแบบก่อนหน้า (แค่โชว์ข้อมูลอ้างอิง)
    hourly = hourly_ch.groupby("hour").agg(
        sales=("sales","sum"),
        orders=("orders","sum"),
        ads=("ads","sum")
    ).reset_index()
    hourly["ROAS"] = hourly["sales"] / hourly["ads"].replace(0, np.nan)

    # ตารางดิบระดับช่องต่อชั่วโมง
    show = (hourly_ch[["hour","channel","ads","orders","sales","sale_ro","ads_ro"]]
            .rename(columns={"ads":"budget(ads)"}).sort_values(["hour","channel"]))
    st.dataframe(show.round(3), use_container_width=True, height=360)

# ============================================================================#
# Channel
# ============================================================================#
elif page == "Channel":
    # ให้เลือกจากรายชื่อช่องทั้งหมด (ไม่ผูกกับตัวกรองด้านซ้าย)
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
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}%",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # snapshot ต่อชั่วโมงของช่องเดียว
    ch_hourly = build_hourly_snapshot(ch_df, by_channel=False)

    # ===== OVERLAY (day-by-day, single metric) =====
    st.markdown("#### Trend overlay by day (channel)")
    metric_label = st.selectbox(
        "Metric to plot (เลือกได้ 1 ค่า)",
        options=["Sales","Orders","Budget (Ads)","sale_ro","ads_ro"],
        index=0, key="ch_metric"
    )
    col_map = {
        "Sales": "sales",
        "Orders": "orders",
        "Budget (Ads)": "ads",
        "sale_ro": "sale_ro",
        "ads_ro": "ads_ro",
    }
    overlay_ch = make_overlay_table(ch_hourly)
    ycol = col_map[metric_label]

    fig_ch = go.Figure()
    for day in sorted(overlay_ch["day"].unique()):
        sub = overlay_ch[overlay_ch["day"] == day].sort_values("hhmm")
        fig_ch.add_trace(go.Scatter(
            x=sub["hhmm"], y=sub[ycol],
            name=str(day), mode="lines+markers"
        ))
    fig_ch.update_layout(
        height=420,
        xaxis_title="Time (HH:MM)",
        yaxis_title=metric_label,
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_ch, use_container_width=True)

    # ตารางต่อชั่วโมงของช่องเดียว
    st.markdown("#### Time series table")
    st.dataframe(
        ch_hourly[["hour","ads","orders","sales","sale_ro","ads_ro"]]
        .rename(columns={"ads":"budget(ads)"}).round(3),
        use_container_width=True, height=360
    )

# ============================================================================#
# Compare (เดิม)
# ============================================================================#
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

    tmp = sub.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    idx = tmp.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    hourly = tmp.loc[idx]

    st.subheader(f"Compare: {', '.join(pick)}")

    # ตาราง KPI เฉลี่ย (เดิม)
    kpis = hourly.groupby("channel").agg(
        ROAS=("sale_ro","mean"),
        AOV=("sales", lambda s: (s.sum() / hourly.loc[s.index, "orders"].sum()) if hourly.loc[s.index, "orders"].sum() else np.nan),
        CPO=("orders", lambda s: (hourly.loc[s.index, "ads"].sum() / s.sum()) if s.sum() else np.nan),
        RPV=("sales", lambda s: (s.sum() / hourly.loc[s.index, "view"].sum()) if hourly.loc[s.index, "view"].sum() else np.nan),
        ORV=("orders", lambda s: (s.sum() / hourly.loc[s.index, "view"].sum()) if hourly.loc[s.index, "view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS","sales","orders","ads"], index=0)

    piv = hourly.pivot_table(index="hour", columns="channel",
                             values=("sale_ro" if met=="ROAS" else met), aggfunc="sum").sort_index()
    rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0

    fig = go.Figure()
    for c in rel.columns:
        if c == base: continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference",
                      legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = hourly.pivot_table(index="hour", columns="channel", values="sale_ro", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
