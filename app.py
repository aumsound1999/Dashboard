# app.py
# Shopee ROAS Dashboard — overview • channel • compare
# Data is read from a Google Sheet (CSV export) provided via Secrets:
#   ROAS_CSV_URL = "https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>"
#
# pip install:
#   streamlit pandas numpy plotly requests

import os, re, io, ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import timedelta

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# =============================================================================
# Helpers: load + parse (เหมือนชุดที่คุณ OK ไว้)
# =============================================================================

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:45
LOW_CREDIT_THRESHOLD = 500  # เครดิตแอดคงเหลือน้อยกว่าเท่านี้ให้แจ้งเตือนพาเนล

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
    """
    "D21 12:4" -> day=21, hour=12, minute=4 ; year/month ใช้ปัจจุบัน
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

def parse_metrics_cell(s: str):
    """
    รับค่าเป็นสตริงตัวเลขคั่นด้วยคอมมา -> list ความยาว 6 (เติม NaN ถ้าไม่ครบ)
    ตัวอย่าง: "2025,12,34,776,22.51,1036"
    แมป: [sales, orders, ads, views, ads_ro, misc]
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
    return nums  # sales, orders, ads, views, ads_ro, misc

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

    df_melt = df_wide.melt(
        id_vars=id_cols, value_vars=time_cols,
        var_name="time_col", value_name="raw"
    )
    # timestamp
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))

    # metrics -> 6 ตัว
    parsed = df_melt["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["sales", "orders", "ads", "view", "ads_ro", "misc"])

    out = pd.concat([df_melt[["timestamp"] + id_cols], V], axis=1).rename(columns={"name": "channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # คำนวณ SaleRO (ระดับ snapshot รวม) — ใช้แบบรวม (ไม่ใช่ Δ) เฉพาะที่ใช้ในตาราง/กราฟบางตัว
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)

    # ให้แน่ใจว่าเป็น numeric
    for c in ["sales","orders","ads","view","ads_ro","misc","SaleRO"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    """เลือกแถวล่าสุดของแต่ละ channel ที่อยู่ในชั่วโมงเดียวกับ at_ts (floor ชั่วโมง)"""
    if df.empty:
        return df
    target_hour = at_ts.floor("H")
    snap = (
        df[df["timestamp"].dt.tz_localize(None).dt.floor("H") == target_hour.tz_localize(None)]
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
    if prev in [0, None] or pd.isna(prev):
        return None
    if curr is None or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

# =============================================================================
# LOW CREDIT PANEL — อ่านจาก df_wide โดยตรง (แทน Data table เดิมใน Overview)
# =============================================================================

# ช่วยตรวจหา column ใน wide
def _wide_is_time_col(c: str) -> bool:
    try:
        return bool(TIME_COL_PATTERN.match(str(c).strip()))
    except Exception:
        return False

def _wide_detect_channel_col(df_wide: pd.DataFrame) -> str:
    for c in df_wide.columns:
        cl = str(c).strip().lower()
        if cl in ("name", "channel", "c_name", "cname"):
            return c
    # fallback: คอลัมน์แรกที่ไม่ใช่เวลา
    for c in df_wide.columns:
        if not _wide_is_time_col(c):
            return c
    return df_wide.columns[0]

def _wide_detect_ads_blob_col(df_wide: pd.DataFrame) -> str | None:
    # หา object col ที่มี 'gmv:' และ 'auto:' และมี [[ ... ]] ข้างใน
    for c in df_wide.columns:
        if df_wide[c].dtype == object:
            sample = " ".join(map(str, df_wide[c].dropna().astype(str).head(5).tolist()))
            if "gmv:" in sample and "auto:" in sample and "[[" in sample:
                return c
    return None

def _wide_list_time_cols(df_wide: pd.DataFrame):
    return [c for c in df_wide.columns if _wide_is_time_col(c)]

def _parse_campaign_blob(s: str):
    """ คืน list ของแคมเปญ: [ [name, budget, spend, orders, views, gmv, roas], ... ] """
    try:
        x = ast.literal_eval(str(s))
        if isinstance(x, tuple) and len(x) >= 3 and isinstance(x[2], list):
            return x[2]
        if isinstance(x, list) and len(x) and isinstance(x[0], list):
            return x
    except Exception:
        pass
    return []

def _count_active_campaigns(blob_val: str) -> int:
    # นิยาม 'เปิดใช้งาน' แบบ conservative: spend > 0
    camps = _parse_campaign_blob(blob_val)
    n = 0
    for it in camps:
        if len(it) >= 3:
            try:
                spend = float(it[2])
                if spend > 0:
                    n += 1
            except Exception:
                pass
    return n

def _extract_credit_from_snapshot(snapshot_val: str) -> float | None:
    """
    snapshot เป็นสตริงตัวเลขคั่นด้วย comma; ตัวสุดท้ายคือเครดิต
    เช่น '2025,12,34,776,22,51,1036' -> 1036
    """
    if not isinstance(snapshot_val, str):
        snapshot_val = str(snapshot_val)
    parts = [p.strip() for p in snapshot_val.split(",") if p.strip() != ""]
    if not parts:
        return None
    try:
        return float(parts[-1])
    except Exception:
        return None

def _parse_ts_from_hdr(hdr: str, tz="Asia/Bangkok"):
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", str(hdr).strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        return pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

def render_low_credit_panel_from_wide(df_wide: pd.DataFrame):
    """อ่านจาก wide: หาช่องที่เครดิต < 500 และมีแคมเปญเปิด (spend>0) แล้ว list ออกมา"""
    if df_wide is None or df_wide.empty:
        st.info("ไม่พบข้อมูล wide")
        return

    ch_col = _wide_detect_channel_col(df_wide)
    ads_col = _wide_detect_ads_blob_col(df_wide)
    time_cols = _wide_list_time_cols(df_wide)

    if not time_cols:
        st.info("ไม่พบคอลัมน์เวลา (เช่น 'D21 12:45') ใน wide")
        return
    if ads_col is None:
        st.info("ไม่พบคอลัมน์ก้อนแอด (ที่มี gmv:/auto:) ใน wide")
        return

    # เรียงคอลัมน์เวลาให้ได้ตัวล่าสุดจริง ๆ
    time_cols_sorted = sorted(time_cols, key=lambda c: _parse_ts_from_hdr(c))
    latest_time_col = time_cols_sorted[-1]

    # เตรียม rows
    work = df_wide[[ch_col, ads_col, latest_time_col]].copy()
    work.columns = ["channel", "ads_blob", "snapshot"]

    work["credit_left"] = work["snapshot"].apply(_extract_credit_from_snapshot)
    work["active_camps"] = work["ads_blob"].apply(_count_active_campaigns)

    filt = (work["credit_left"].notna()) & (work["credit_left"] < LOW_CREDIT_THRESHOLD) & (work["active_camps"] > 0)
    warn = work.loc[filt, ["channel", "credit_left", "active_camps"]]\
               .sort_values(["credit_left", "active_camps"], ascending=[True, False])

    st.markdown("### Advertising credits are low")
    st.caption("แสดงเฉพาะช่องที่เครดิตคงเหลือน้อยกว่า 500 และมีแคมเปญที่เปิดใช้งานอยู่ (ตาม spend > 0)")

    if warn.empty:
        st.success("ทุกช่องมีเครดิตเพียงพอ หรือยังไม่มีแคมเปญที่เปิดอยู่ 😊")
        return

    st.dataframe(
        warn.rename(columns={"channel": "ช่อง", "credit_left": "เครดิตคงเหลือ", "active_camps": "แคมเปญที่เปิด"})
            .style.format({"เครดิตคงเหลือ": lambda x: f"{int(x):,}"}),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# UI: Data refresh header (Reload only)
# =============================================================================

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

# =============================================================================
# Sidebar filters — default 3 days, Channels [All]
# =============================================================================

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

# กรองช่วงวันที่ + ช่อง (ใช้ d สำหรับกราฟ/ตารางที่ต้องการช่วง)
mask = (
    (df_long["timestamp"] >= start_ts)
    & (df_long["timestamp"] <= end_ts)
    & (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

# แสดงหัวเรื่อง + last refresh
st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Overview
# =============================================================================
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
    C[1].metric("Orders", f"{cur['Orders']:,.0f}%",
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

    # ===== Trend by hour (Sales/Orders/ROAS) =====
    # เลือกสแนปช็อตล่าสุดของแต่ละชั่วโมง/ช่อง
    tmp = d.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    idx = tmp.sort_values("timestamp").groupby(["channel", "hour"]).tail(1).index
    hourly = tmp.loc[idx].copy()

    trend = hourly.groupby("hour").agg(
        sales=("sales", "sum"),
        orders=("orders", "sum"),
        ads=("ads", "sum")
    ).reset_index()
    trend["ROAS"] = trend["sales"] / trend["ads"].replace(0, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["sales"], name="Sales", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["orders"], name="Orders", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["ROAS"], name="ROAS", mode="lines+markers", yaxis="y3"))
    fig.update_layout(
        height=380,
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="ROAS", overlaying="y", side="right", position=1.0),
        legend=dict(orientation="h", y=-0.25),
    )
    st.markdown("#### Trend by hour (Sales/Orders/ROAS)")
    st.plotly_chart(fig, use_container_width=True)

    # ===== Prime hours heatmap =====
    st.markdown("#### Prime hours heatmap")
    tmp = hourly.copy()
    tmp["day"] = tmp["hour"].dt.date
    tmp["h"] = tmp["hour"].dt.hour
    heat = tmp.groupby(["day", "h"]).agg(val=("sales", "sum")).reset_index()
    pivot = heat.pivot(index="day", columns="h", values="val").sort_index(ascending=False)
    fig_h = px.imshow(pivot, aspect="auto", labels=dict(x="Hour", y="Day", color="Sales"))
    st.plotly_chart(fig_h, use_container_width=True)

    # ===== แทนที่ Data table เดิมด้วย Low-credit panel =====
    st.markdown("---")
    render_low_credit_panel_from_wide(wide)   # <<<< ใหม่
    st.markdown("---")

# =============================================================================
# Channel
# =============================================================================
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
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # line หลายแกนสำหรับช่องเดียว
    tmp = ch_df.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    idx = tmp.sort_values("timestamp").groupby("hour").tail(1).index
    ch_hourly = tmp.loc[idx].sort_values("hour")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ch_hourly["hour"], y=ch_hourly["sales"], name="Sales", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ch_hourly["hour"], y=ch_hourly["orders"], name="Orders", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=ch_hourly["hour"], y=ch_hourly["ads"], name="Ads", mode="lines+markers", yaxis="y3"))
    fig.add_trace(go.Scatter(x=ch_hourly["hour"], y=ch_hourly["SaleRO"], name="sale_ro", mode="lines+markers", yaxis="y4"))
    fig.update_layout(
        height=420,
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="Ads", overlaying="y", side="right", position=1.0),
        yaxis4=dict(title="sale_ro", overlaying="y", side="right", position=0.95),
        legend=dict(orientation="h", y=-0.28),
    )
    st.markdown("#### Multi-axis line")
    st.plotly_chart(fig, use_container_width=True)

    # ตารางต่อชั่วโมงของช่องเดียว (คงไว้ตามเดิมที่คุณใช้ได้ปกติ)
    st.markdown("#### Time series table (hourly)")
    st.dataframe(
        ch_hourly[["hour", "ads", "orders", "sales", "SaleRO", "ads_ro"]]
        .rename(columns={"ads": "budget(ads)"}).round(3),
        use_container_width=True,
        height=360,
    )

# =============================================================================
# Compare
# =============================================================================
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

    # สแนปช็อตล่าสุดต่อชั่วโมง/ช่อง
    tmp = sub.copy()
    tmp["hour"] = tmp["timestamp"].dt.floor("H")
    idx = tmp.sort_values("timestamp").groupby(["channel", "hour"]).tail(1).index
    hourly = tmp.loc[idx]

    st.subheader(f"Compare: {', '.join(pick)}")

    # ตาราง KPI เฉลี่ย
    kpis = hourly.groupby("channel").agg(
        ROAS=("SaleRO", "mean"),
        AOV=("sales", lambda s: (s.sum() / hourly.loc[s.index, "orders"].sum()) if hourly.loc[s.index, "orders"].sum() else np.nan),
        CPO=("orders", lambda s: (hourly.loc[s.index, "ads"].sum() / s.sum()) if s.sum() else np.nan),
        RPV=("sales", lambda s: (s.sum() / hourly.loc[s.index, "view"].sum()) if hourly.loc[s.index, "view"].sum() else np.nan),
        ORV=("orders", lambda s: (s.sum() / hourly.loc[s.index, "view"].sum()) if hourly.loc[s.index, "view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS", "sales", "orders", "ads"], index=0)

    piv = hourly.pivot_table(index="hour", columns="channel", values=("SaleRO" if met == "ROAS" else met), aggfunc="sum").sort_index()
    rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0

    fig = go.Figure()
    for c in rel.columns:
        if c == base:
            continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference", legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = hourly.pivot_table(index="hour", columns="channel", values="SaleRO", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
