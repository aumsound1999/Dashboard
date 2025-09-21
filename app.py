# Shopee ROAS Dashboard — overlay-by-day (1-metric), Overview & Channel
# Requirements (requirements.txt):
# streamlit==1.37.1
# pandas==2.2.2
# numpy==1.26.4
# plotly==5.24.0
# requests==2.32.3

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Shopee ROAS Dashboard", layout="wide")

# -----------------------------
# Helpers for parsing sheet
# -----------------------------
# headers like D21 15:55
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")

def is_time_col(col: str) -> bool:
    return bool(TIME_COL_PATTERN.match(col.strip()))

def parse_timestamp_header(hdr: str, tz="Asia/Bangkok") -> pd.Timestamp:
    """
    "D21 15:55" -> today_year, today_month, day=21, hour=15, minute=55
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

def str_to_tuple6(s: str):
    """
    Expect "budget,user,order,view,sale,ro" (butบางครั้งมีตัวอักษร/ช่องว่าง)
    คืนค่า 6 ค่า (float หรือ NaN)
    """
    if not isinstance(s, str) or not re.search(r"\d", s):
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
    """
    wide -> long
    columns:
      - 'name' (channel)
      - many time cols like 'D21 15:55'
      - possibly meta columns (ignored)
    each value in time cols => "budget,user,order,view,sale,ro"
    """
    cols = list(df_wide.columns)
    id_cols, time_cols = [], []
    for c in cols:
        if c.strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
        else:
            pass
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")

    # parse timestamp from header
    df_melt["timestamp"] = df_melt["time_col"].apply(parse_timestamp_header)
    df_melt = df_melt.dropna(subset=["timestamp"]).reset_index(drop=True)

    # parse metrics to columns
    metrics = df_melt["metrics"].apply(str_to_tuple6)
    md = pd.DataFrame(metrics.tolist(),
                      columns=["budget","user","order","view","sale","ads_ro"])  # ro = ads_ro จากชีท
    out = pd.concat([df_melt[["timestamp","name"]], md], axis=1).rename(columns={"name":"channel"})

    # Derived KPIs
    out["ROAS"] = out["sale"] / out["budget"].replace(0, np.nan)       # sale_ro
    out["ORV"]  = out["order"] / out["view"].replace(0, np.nan)
    out["AOV"]  = out["sale"] / out["order"].replace(0, np.nan)
    out["CPO"]  = out["budget"] / out["order"].replace(0, np.nan)
    out["RPV"]  = out["sale"] / out["view"].replace(0, np.nan)

    return out

# -----------------------------
# Data loading (URL from secrets)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)  # cache 5 นาที
def fetch_wide_csv(url: str) -> pd.DataFrame:
    import requests
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

@st.cache_data(ttl=300, show_spinner=False)
def build_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    return long_from_wide(df_wide)

def load_data():
    url = os.environ.get("ROAS_CSV_URL", "")
    if not url:
        st.error("ไม่พบ ROAS_CSV_URL ใน Secrets / Environment")
        st.stop()
    df_wide = fetch_wide_csv(url)
    df_long = build_long(df_wide)
    return df_long

# -----------------------------
# Aggregations used in pages
# -----------------------------
def latest_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    เก็บ snapshot ล่าสุดในแต่ละชั่วโมงของแต่ละ channel
    """
    x = df.copy()
    x["hour"] = x["timestamp"].dt.floor("H")
    idx = x.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return x.loc[idx].reset_index(drop=True)

def date_filters_defaults(df: pd.DataFrame):
    tz = "Asia/Bangkok"
    max_ts = df["timestamp"].max()
    if pd.isna(max_ts):
        today = pd.Timestamp.now(tz=tz).date()
    else:
        today = max_ts.tz_convert(tz).date()
    start = today - timedelta(days=2)  # รวมวันนี้ = 3 วัน
    return start, today

def compute_kpi_totals(df):
    res = {}
    res["Sales"]  = df["sale"].sum()
    res["Orders"] = df["order"].sum()
    res["Budget"] = df["budget"].sum()
    res["ROAS"]   = (df["sale"].sum()/df["budget"].sum()) if df["budget"].sum()!=0 else np.nan
    res["AOV"]    = (df["sale"].sum()/df["order"].sum()) if df["order"].sum()!=0 else np.nan
    res["CPO"]    = (df["budget"].sum()/df["order"].sum()) if df["order"].sum()!=0 else np.nan
    res["RPV"]    = (df["sale"].sum()/df["view"].sum()) if df["view"].sum()!=0 else np.nan
    res["ORV"]    = (df["order"].sum()/df["view"].sum()) if df["view"].sum()!=0 else np.nan
    return res

def pct_delta(curr, prev):
    if prev is None or pd.isna(prev) or prev==0:
        return None
    if pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

def hr_min(series_ts):
    # format HH:MM
    return series_ts.dt.strftime("%H:%M")

def hourly_overlay(df_hourly, channels, date_from, date_to, metric_col):
    """
    สร้างข้อมูล overlay-by-day สำหรับ metric เดียว
    - เลือกเฉพาะ channels ที่เลือก
    - ตัดช่วงวันที่ตาม date_from .. date_to (รวม)
    คืนค่า: pivot (index=HH:MM, columns=day, values=metric)
    และ df_table ใช้โชว์ตาราง
    """
    d = df_hourly[df_hourly["channel"].isin(channels)].copy()

    tz = "Asia/Bangkok"
    start_ts = pd.Timestamp(date_from).tz_localize(tz)
    end_ts   = pd.Timestamp(date_to).tz_localize(tz) + pd.Timedelta(hours=23, minutes=59)

    d = d[(d["hour"]>=start_ts) & (d["hour"]<=end_ts)].copy()
    d["day"]  = d["hour"].dt.date
    d["hhmm"] = d["hour"].dt.strftime("%H:%M")

    # รวม across channels แต่ละ (day,hhmm)
    agg = d.groupby(["day","hhmm"], as_index=False).agg(
        sale=("sale","sum"),
        order=("order","sum"),
        budget=("budget","sum"),
        sale_ro=("ROAS","mean"),
        ads_ro=("ads_ro","mean"),
        view=("view","sum")
    )
    # เลือกคอลัมน์ metric
    metric_map = {
        "Sales":"sale",
        "Orders":"order",
        "Budget":"budget",
        "ROAS (sale_ro)":"sale_ro",
        "Ads_RO (ads_ro)":"ads_ro",
    }
    mcol = metric_map[metric_col]
    # pivot สำหรับกราฟ
    piv = agg.pivot_table(index="hhmm", columns="day", values=mcol, aggfunc="sum").sort_index()
    # ตารางใช้ข้อมูลเดียวกัน (แปลงเป็น long เพื่อไฮไลต์)
    table = agg[["day","hhmm", mcol]].rename(columns={mcol: metric_col}).sort_values(["day","hhmm"])
    return piv, table

def draw_overlay_chart(pivot_df, metric_label, ytitle=None):
    fig = go.Figure()
    # เส้นละวัน
    for day in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index, y=pivot_df[day],
            mode="lines+markers",
            name=str(day),
            connectgaps=False
        ))
    fig.update_layout(
        height=380,
        xaxis_title="Time (HH:MM)",
        yaxis_title=ytitle or metric_label,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10,r=10,t=10,b=10)
    )
    return fig

def style_table(df, value_col):
    # pandas Styler for gradient on the value column
    try:
        styled = (df.style
                  .background_gradient(cmap="YlOrRd", subset=[value_col])
                  .format({value_col: lambda x: f"{x:,.2f}" if isinstance(x,(int,float,np.floating)) else x}))
        return styled
    except Exception:
        return df

# -----------------------------
# App
# -----------------------------
st.title("Shopee ROAS Dashboard")

# Load once (cache 5 นาที)
if "df_long" not in st.session_state:
    st.session_state["df_long"] = load_data()
df_long = st.session_state["df_long"]

# Snapshot hourly (ล่าสุดต่อชั่วโมง/ช่อง)
df_hourly_all = latest_per_hour(df_long)

# Controls (sidebar)
st.sidebar.header("Filters")
# date range default = 3 วัน
start_default, end_default = date_filters_defaults(df_long)
d1, d2 = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(start_default, end_default)
)
if isinstance(d1, (list,tuple)):  # guard old streamlit behavior
    d1, d2 = d1

# Channel list (มี [All])
channels_all = sorted(df_long["channel"].dropna().unique().tolist())
ch_pick = st.sidebar.multiselect("Channels (เลือก All ได้)",
                                 options=["[All]"] + channels_all,
                                 default=["[All]"])
if "[All]" in ch_pick or len(ch_pick)==0:
    sel_channels = channels_all
else:
    sel_channels = ch_pick

# เมนูหน้า
page = st.sidebar.radio("Page", ["Overview", "Channel"], index=0)

# ปุ่ม reload
col_t, col_btn = st.columns([4,1])
with col_btn:
    if st.button("Reload", type="primary"):
        st.cache_data.clear()
        st.session_state["df_long"] = load_data()
        st.rerun()
st.caption(f"Last refresh: {pd.Timestamp.now('Asia/Bangkok').strftime('%Y-%m-%d %H:%M:%S')}")

# ---- Baseline (yesterday) สำหรับ KPI
tz = "Asia/Bangkok"
start_ts = pd.Timestamp(d1).tz_localize(tz)
end_ts   = pd.Timestamp(d2).tz_localize(tz) + pd.Timedelta(hours=23,minutes=59)
mask_curr = (df_hourly_all["channel"].isin(sel_channels)) & \
            (df_hourly_all["hour"]>=start_ts) & (df_hourly_all["hour"]<=end_ts)
curr_df = df_hourly_all.loc[mask_curr].copy()

y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts - pd.Timedelta(days=1)
mask_y  = (df_hourly_all["channel"].isin(sel_channels)) & \
          (df_hourly_all["hour"]>=y_start) & (df_hourly_all["hour"]<=y_end)
y_df = df_hourly_all.loc[mask_y].copy()

# KPI (รวมทุกช่องที่เลือก)
tot = compute_kpi_totals(curr_df)
base_tot = compute_kpi_totals(y_df)

# แสดง KPI (แถบด้านบน)
kpi_cols = st.columns(8)
kpi_labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
for i, lab in enumerate(kpi_labels):
    curr = tot.get(lab, np.nan)
    prev = base_tot.get(lab, np.nan)
    delta = pct_delta(curr, prev)
    with kpi_cols[i]:
        if lab in ["ROAS","AOV","CPO","RPV","ORV"]:
            show_val = "-" if pd.isna(curr) else f"{curr:,.2f}"
        else:
            show_val = "-" if pd.isna(curr) else f"{curr:,.0f}"
        st.metric(lab, show_val, None if delta is None else f"{delta:+.1f}%")

# -----------------------------
# OVERVIEW PAGE
# -----------------------------
if page == "Overview":
    st.subheader("Trend overlay by day")

    metric_label = st.selectbox(
        "Metric to plot (1 ค่าเท่านั้น)",
        ["Sales","Orders","Budget","ROAS (sale_ro)","Ads_RO (ads_ro)"],
        index=0, key="ov_metric"
    )

    piv, tbl = hourly_overlay(df_hourly_all, sel_channels, d1, d2, metric_label)
    fig = draw_overlay_chart(piv, metric_label, ytitle=metric_label)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Data (hourly latest snapshot aggregated)**")
    st.dataframe(style_table(tbl, metric_label), use_container_width=True)

# -----------------------------
# CHANNEL PAGE
# -----------------------------
else:
    st.subheader("Channel view")

    # dropdown ช่อง (จากทั้งหมด ไม่ได้ตาม filter)
    ch = st.selectbox("Pick one channel (ทั้งหมดในชีต)", options=channels_all, index=0)
    # metric
    metric_label = st.selectbox(
        "Metric to plot (1 ค่าเท่านั้น)",
        ["Sales","Orders","Budget","ROAS (sale_ro)","Ads_RO (ads_ro)"],
        index=0, key="ch_metric"
    )

    piv, tbl = hourly_overlay(df_hourly_all, [ch], d1, d2, metric_label)
    fig = draw_overlay_chart(piv, metric_label, ytitle=metric_label)
    st.plotly_chart(fig, use_container_width=True)

    # KPI ของช่องเดียว + เทียบเมื่อวาน
    ch_curr = curr_df[curr_df["channel"]==ch]
    ch_y    = y_df[y_df["channel"]==ch]
    t1 = compute_kpi_totals(ch_curr); t0 = compute_kpi_totals(ch_y)
    cols = st.columns(8)
    for i, lab in enumerate(kpi_labels):
        curr = t1.get(lab, np.nan); prev = t0.get(lab, np.nan)
        delta = pct_delta(curr, prev)
        with cols[i]:
            show_val = "-" if pd.isna(curr) else (f"{curr:,.2f}" if lab in ["ROAS","AOV","CPO","RPV","ORV"] else f"{curr:,.0f}")
            st.metric(lab, show_val, None if delta is None else f"{delta:+.1f}%")

    st.markdown("**Data (hourly latest snapshot aggregated for this channel)**")
    st.dataframe(style_table(tbl, metric_label), use_container_width=True)
