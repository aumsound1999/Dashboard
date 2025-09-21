# app.py
# Shopee ROAS Dashboard — overview • channel • compare
# Data is read from a Google Sheet (CSV export) provided via Secrets:
#   ROAS_CSV_URL = "https://docs.google.com/spreadsheets/d/<ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>"

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g. D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr.strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        return pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

def parse_metrics_cell(s: str):
    # "2025,12,34,776,22.51,1036" -> v0..v5
    if not isinstance(s, str) or not re.search(r"\d", s):
        return [np.nan]*6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p!=""]
    nums = []
    for p in parts[:6]:
        try: nums.append(float(p))
        except: nums.append(np.nan)
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
    return pd.read_csv(io.StringIO(text))

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
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
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))

    parsed = df_melt["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([df_melt[["timestamp"]+id_cols], V], axis=1).rename(columns={"name":"channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # map v -> business metrics (ปรับให้ตรงกับชีตของคุณแล้ว)
    out["sales"]  = pd.to_numeric(out["v0"], errors="coerce")  # 2025
    out["orders"] = pd.to_numeric(out["v1"], errors="coerce")  # 12
    out["ads"]    = pd.to_numeric(out["v2"], errors="coerce")  # 34 (budget/ads)
    out["view"]   = pd.to_numeric(out["v3"], errors="coerce")  # 776
    out["ads_ro"] = pd.to_numeric(out["v4"], errors="coerce")  # 22.51
    out["misc"]   = pd.to_numeric(out["v5"], errors="coerce")  # 1036

    # main ROAS (sales / ads)
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def pick_snapshot_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """เลือกสแนปช็อตล่าสุดของแต่ละ (channel, hour)"""
    if df.empty: return df
    tmp = df.copy()
    tmp["hour"]  = tmp["timestamp"].dt.floor("H")
    tmp["hhmm"]  = tmp["hour"].dt.strftime("%H:%M")
    tmp["day"]   = tmp["hour"].dt.date
    idx = tmp.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return tmp.loc[idx].copy()

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    s = snap["sales"].sum()
    o = snap["orders"].sum()
    a = snap["ads"].sum()
    sale_ro = (s/a) if a else np.nan
    ads_ro_avg = snap.loc[snap["ads_ro"]>0,"ads_ro"].mean()
    return dict(Sales=s, Orders=o, Ads=a, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr): return None
    return (curr - prev) * 100.0 / prev

# --- plotting helpers --------------------------------------------------------
HOURS_00_23 = pd.date_range("00:00","23:00",freq="H").strftime("%H:%M")

def overlay_by_day(hourly_latest: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    สรุปค่า metric ต่อวัน-ต่อชั่วโมง (hhmm) แล้ว reindex ให้ครบทุกชั่วโมง
    metric_key: 'sales'|'orders'|'ads'|'sale_ro'|'ads_ro'
    """
    # แก้ mapping sale_ro -> SaleRO (ชื่อจริงใน df)
    metric_map = {
        "sales":"sales",
        "orders":"orders",
        "ads":"ads",
        "sale_ro":"SaleRO",   # <— สำคัญ
        "ads_ro":"ads_ro",
    }
    if metric_key not in metric_map:
        raise KeyError(f"Unknown metric: {metric_key}")
    col = metric_map[metric_key]

    # aggregate per day-hhmm
    g = hourly_latest.groupby(["day","hhmm"]).agg(
        sales=("sales","sum"),
        orders=("orders","sum"),
        ads=("ads","sum"),
        ads_ro=("ads_ro", lambda s: s[s>0].mean())
    ).reset_index()
    g["SaleRO"] = g["sales"]/g["ads"].replace(0,np.nan)

    # เลือกคอลัมน์ที่ต้องการ
    g = g[["day","hhmm", col]].rename(columns={col:"value"})

    # clamp เฉพาะตอน plot
    if metric_key in ("sale_ro","ads_ro"):
        g["value"] = g["value"].clip(lower=0, upper=50)

    # เติมช่องว่างให้ครบ 00:00-23:00
    frames = []
    for d in sorted(g["day"].unique()):
        sub = g[g["day"]==d].set_index("hhmm").reindex(HOURS_00_23).rename_axis("hhmm").reset_index()
        sub["day"] = d
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out  # columns: day, hhmm, value

def plot_overlay(out_df: pd.DataFrame, y_label: str, title: str=""):
    fig = go.Figure()
    for d in sorted(out_df["day"].unique()):
        sub = out_df[out_df["day"]==d]
        fig.add_trace(go.Scatter(
            x=sub["hhmm"], y=sub["value"], mode="lines+markers", name=str(d)
        ))
    fig.update_layout(
        height=420,
        xaxis_title="Time (HH:MM)",
        yaxis_title=y_label,
        legend=dict(orientation="h", y=-0.25),
        title=title
    )
    return fig

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

# Reload button
if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear(); load_wide_df.clear(); build_long.clear()
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

# date range (default 3 days)
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
if isinstance(d1,(list,tuple)): d1, d2 = d1
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

# channels
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# filter data
mask = (
    (df_long["timestamp"]>=start_ts) & (df_long["timestamp"]<=end_ts) &
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

    # snapshots (current & yesterday same hour)
    hl = pick_snapshot_hourly(d)
    cur_ts = d["timestamp"].max()
    cur_hour = cur_ts.floor("H")
    y_hour   = cur_hour - pd.Timedelta(days=1)

    cur_snap = hl[hl["hour"]==cur_hour]
    y_snap   = hl[hl["hour"]==y_hour]

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

    # Overlay by day (one metric)
    st.markdown("### Trend overlay by day")
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                          options=["sales","orders","ads","sale_ro","ads_ro"], index=0, key="ov_metric")
    try:
        ov = overlay_by_day(hl, metric)
        ylabel = metric
        fig = plot_overlay(ov, y_label=ylabel)
        st.plotly_chart(fig, use_container_width=True)
    except KeyError as e:
        st.error(f"Plot failed: {e}")

    # Heatmap (sales)
    st.markdown("### Prime hours heatmap")
    heat = hl.groupby(["day","hhmm"]).agg(val=("sales","sum")).reset_index()
    pivot = heat.pivot(index="day", columns="hhmm", values="val").reindex(columns=HOURS_00_23)
    fig_h = px.imshow(pivot, aspect="auto", labels=dict(x="Hour", y="Day", color="Sales"))
    st.plotly_chart(fig_h, use_container_width=True)

    # (ทำให้ปลอดภัย ถ้าอยากแสดงตาราง) — ไม่บังคับ
    st.markdown("### Data (hourly latest snapshot per channel)")
    safe_cols = ["day","hhmm","channel","ads","orders","sales","SaleRO","ads_ro"]
    show = hl.copy()[safe_cols].rename(columns={"SaleRO":"sale_ro"})
    st.dataframe(show.round(3), use_container_width=True, height=360)

# -----------------------------------------------------------------------------
# Channel
# -----------------------------------------------------------------------------
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = d[d["channel"]==ch].copy()
    if ch_df.empty:
        st.warning("No data for this channel.")
        st.stop()

    hl = pick_snapshot_hourly(ch_df)
    cur_ts = ch_df["timestamp"].max()
    cur_hour = cur_ts.floor("H")
    y_hour   = cur_hour - pd.Timedelta(days=1)
    cur_snap = hl[hl["hour"]==cur_hour]
    y_snap   = hl[hl["hour"]==y_hour]

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
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                          options=["sales","orders","ads","sale_ro","ads_ro"], index=0, key="ch_metric")
    try:
        ov = overlay_by_day(hl, metric)
        fig = plot_overlay(ov, y_label=metric)
        st.plotly_chart(fig, use_container_width=True)
    except KeyError as e:
        st.error(f"Plot failed: {e}")

    st.markdown("### Time series table")
    st.dataframe(
        hl[["hour","ads","orders","sales","SaleRO","ads_ro"]]
          .rename(columns={"SaleRO":"sale_ro", "ads":"budget(ads)"})
          .round(3),
        use_container_width=True, height=360
    )

# -----------------------------------------------------------------------------
# Compare
# -----------------------------------------------------------------------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels,
                          default=all_channels[:2], max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels."); st.stop()
    sub = d[d["channel"].isin(pick)].copy()
    if sub.empty:
        st.warning("No data for selected channels in range."); st.stop()

    hl = pick_snapshot_hourly(sub)
    st.subheader(f"Compare: {', '.join(pick)}")

    kpis = hl.groupby("channel").agg(
        ROAS=("sales", lambda s: s.sum()/hl.loc[s.index,"ads"].sum() if hl.loc[s.index,"ads"].sum() else np.nan),
        AOV=("sales", lambda s: s.sum()/hl.loc[s.index,"orders"].sum() if hl.loc[s.index,"orders"].sum() else np.nan),
        CPO=("orders", lambda s: hl.loc[s.index,"ads"].sum()/s.sum() if s.sum() else np.nan),
        RPV=("sales", lambda s: s.sum()/hl.loc[s.index,"view"].sum() if hl.loc[s.index,"view"].sum() else np.nan),
        ORV=("orders", lambda s: s.sum()/hl.loc[s.index,"view"].sum() if hl.loc[s.index,"view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met  = st.selectbox("Metric", options=["ROAS","sales","orders","ads"], index=0)

    piv = hl.pivot_table(index="hour", columns="channel",
                         values=("sales" if met!="ROAS" else "SaleRO"),
                         aggfunc=("sum" if met!="ROAS" else "mean")).sort_index()
    if met=="ROAS": piv = piv.clip(upper=50)
    rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0

    fig = go.Figure()
    for c in rel.columns:
        if c == base: continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference",
                      legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = hl.pivot_table(index="hour", columns="channel", values="SaleRO", aggfunc="mean").sort_index().clip(upper=50)
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
