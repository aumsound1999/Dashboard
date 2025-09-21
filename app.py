# app.py
# Shopee ROAS Dashboard — Overview • Channel • Compare (safe hourly)
# Env: ROAS_CSV_URL = "https://docs.google.com/spreadsheets/.../gviz/tq?tqx=out:csv&sheet=<SHEET>"

import os, io, re
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers: parsing
# -----------------------------------------------------------------------------
HDR_RE = re.compile(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$")  # e.g. D21 23:49

def is_time_col(col: str) -> bool:
    if not isinstance(col, str): return False
    return HDR_RE.match(col.strip()) is not None

def parse_hdr_timestamp(col: str, tz="Asia/Bangkok") -> pd.Timestamp:
    m = HDR_RE.match(str(col).strip())
    if not m: return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        return pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

def parse_metrics_cell(s: str, n=6):
    """
    รับสตริงรูป "2025,12,34,776,22.51,1036" => [v0..v5]
    ถ้าเละ/ไม่มีตัวเลข -> NaN ทั้งชุด
    """
    if not isinstance(s, str) or not re.search(r"\d", s):
        return [np.nan]*n
    s = re.sub(r"[^0-9\.\-,]", "", s)  # keep 0-9 . - ,
    parts = [p for p in s.split(",") if p != ""]
    vals = []
    for p in parts[:n]:
        try: vals.append(float(p))
        except: vals.append(np.nan)
    while len(vals) < n: vals.append(np.nan)
    return vals

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_csv_text() -> str:
    url = os.environ.get("ROAS_CSV_URL", "")
    if not url: raise RuntimeError("Missing env ROAS_CSV_URL")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=600, show_spinner=True)
def load_wide_df() -> pd.DataFrame:
    txt = fetch_csv_text()
    return pd.read_csv(io.StringIO(txt))

# -----------------------------------------------------------------------------
# Transform wide->long (and derive hourly-safe table)
# -----------------------------------------------------------------------------
def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # Identify id/time columns
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns like 'D21 23:49' found.")

    # Melt
    m = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                     var_name="time_col", value_name="raw").rename(columns={"name":"channel"})

    # Timestamp
    m["timestamp"] = m["time_col"].apply(lambda x: parse_hdr_timestamp(x, tz=tz))
    m = m.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Parse metrics from 'raw' -> v0..v5
    parsed = m["raw"].apply(lambda s: parse_metrics_cell(s, n=6))
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([m[["channel","timestamp"]], V], axis=1)

    # Mapping (ปรับให้ตรงชีทจริงได้):
    # v0: sales cum, v1: orders cum, v2: ads/budget cum, v3: views cum (ไม่ใช้),
    # v4: 'ads_ro raw?' (ไม่ใช้ตรง ๆ), v5: misc
    out["sales_cum"]   = pd.to_numeric(out["v0"], errors="coerce")
    out["orders_cum"]  = pd.to_numeric(out["v1"], errors="coerce")
    out["ads_cum"]     = pd.to_numeric(out["v2"], errors="coerce")

    # ถ้ามี "ยอดขายจากโฆษณาแบบสะสม" (optional) ใส่ชื่อคีย์ตรงนี้ (เช่น v5 หรือคอลัมน์อื่น)
    # ถ้าไม่มีจะ fallback ใช้ยอดขายรวมเวลาคิด ads_ro_hour
    out["sales_from_ads_cum"] = pd.to_numeric(out.get("v5", np.nan), errors="coerce")

    out = out.drop(columns=["v0","v1","v2","v3","v4","v5","raw","time_col"], errors="ignore")
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    return long_from_wide(df_wide)

# -----------------------------------------------------------------------------
# Hourly-safe snapshot table
# -----------------------------------------------------------------------------
def build_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """เลือก snapshot ล่าสุดของแต่ละชั่วโมง/ช่อง และคำนวณค่า 'ต่างชั่วโมง' ภายในวัน"""
    if df_long.empty:
        return df_long.copy()

    h = df_long.copy()
    h["hour_key"] = h["timestamp"].dt.floor("H")

    # เลือก snapshot ล่าสุดของทุก (channel, hour_key)
    idx = h.sort_values("timestamp").groupby(["channel", "hour_key"]).tail(1).index
    h = h.loc[idx].copy()

    # ช่วยสำหรับกรุ๊ปภายในวัน
    h["day"] = h["hour_key"].dt.date
    h["hour"] = h["hour_key"].dt.hour
    h = h.sort_values(["channel", "day", "hour_key"])

    # ให้แน่ใจว่าเป็นตัวเลข
    for c in ["sales_cum", "orders_cum", "ads_cum", "sales_from_ads_cum"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")

    # diff แบบ index-align ด้วย groupby(...)[col].diff()
    def _diff_nonneg(colname, outname):
        if colname not in h.columns:
            h[outname] = 0.0
            return
        d = h.groupby(["channel", "day"], sort=False)[colname].diff()
        d = d.fillna(0).clip(lower=0)
        h[outname] = d

    _diff_nonneg("sales_cum",  "sales_hour")
    _diff_nonneg("orders_cum", "orders_hour")
    _diff_nonneg("ads_cum",    "ads_hour")
    _diff_nonneg("sales_from_ads_cum", "sales_ads_hour")

    # ROAS รายชั่วโมง (limit 50)
    with np.errstate(divide="ignore", invalid="ignore"):
        h["sale_ro_hour"] = np.where(h["ads_hour"] > 0, h["sales_hour"] / h["ads_hour"], np.nan)

        # ถ้าไม่มียอดขายจากแอด ให้ fallback เป็นยอดขายรวมรายชั่วโมง
        base_sales_ads = np.where(h["sales_ads_hour"].notna(), h["sales_ads_hour"], h["sales_hour"])
        h["ads_ro_hour"] = np.where(h["ads_hour"] > 0, base_sales_ads / h["ads_hour"], np.nan)

    h["sale_ro_hour"] = h["sale_ro_hour"].clip(upper=50)
    h["ads_ro_hour"]  = h["ads_ro_hour"].clip(upper=50)

    return h


# -----------------------------------------------------------------------------
# KPI helpers
# -----------------------------------------------------------------------------
def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty: return df
    target = at_ts.floor("H")
    snap = (
        df[df["hour_key"]==target]
        .sort_values("timestamp")
        .groupby("channel")
        .tail(1)
    )
    return snap

def safe_mean_pos(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    s = s[s>0]
    return s.mean() if len(s) else np.nan

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales_cum"].sum()   # สะสม ณ ชั่วโมงนั้น
    orders = snap["orders_cum"].sum()
    ads    = snap["ads_cum"].sum()
    sale_ro = (sales/ads) if ads>0 else np.nan
    ads_ro_avg = safe_mean_pos(snap.get("ads_ro_hour", pd.Series(dtype=float)))  # ใช้ค่า h/hr >0 เฉลี่ย
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(a, b):
    try:
        if b is None or pd.isna(b) or b==0: return None
        if a is None or pd.isna(a): return None
        v = (a-b)*100.0/abs(b)
        if not np.isfinite(v): return None
        return v
    except: return None

# -----------------------------------------------------------------------------
# Plot builders
# -----------------------------------------------------------------------------
METRIC_MAP = {
    "sales":   "sales_hour",
    "orders":  "orders_hour",
    "ads":     "ads_hour",
    "sale_ro": "sale_ro_hour",
    "ads_ro":  "ads_ro_hour",
}

def build_overlay_by_day(hourly: pd.DataFrame, metric_key: str, title: str):
    """ซ้อนเส้นรายวัน: X=hour(0..23), Y=metric ที่เลือก; หนึ่งสี = 1 วัน"""
    if hourly.empty:
        st.info("No data to plot."); return

    use_col = METRIC_MAP[metric_key]
    cols_needed = ["day","hour", use_col]
    if not all(c in hourly.columns for c in cols_needed):
        st.info("Missing columns for plot."); return

    agg = hourly.groupby(["day","hour"], as_index=False)[use_col].sum()

    # สร้างแกน hour 0..23 ทุกวัน (กันรู)
    days = agg["day"].sort_values().unique()
    frame = pd.MultiIndex.from_product([days, range(24)], names=["day","hour"])
    agg = agg.set_index(["day","hour"]).reindex(frame).reset_index()
    agg[use_col] = agg[use_col].fillna(0)

    fig = go.Figure()
    for d, g in agg.groupby("day", sort=True):
        fig.add_trace(
            go.Scatter(
                x=g["hour"], y=g[use_col],
                mode="lines+markers",
                name=str(d)
            )
        )
    fig.update_layout(
        height=360,
        title=title,
        xaxis_title="Hour (0–23)",
        yaxis_title=metric_key
    )
    st.plotly_chart(fig, use_container_width=True)

def build_prime_heatmap(hourly: pd.DataFrame, metric_key: str):
    if hourly.empty:
        return
    use_col = METRIC_MAP[metric_key]
    heat = hourly.groupby(["day","hour"], as_index=False)[use_col].sum()
    days = heat["day"].sort_values().unique()
    frame = pd.MultiIndex.from_product([days, range(24)], names=["day","hour"])
    heat = heat.set_index(["day","hour"]).reindex(frame).reset_index()
    heat[use_col] = heat[use_col].fillna(0)

    pivot = heat.pivot(index="day", columns="hour", values=use_col)
    fig = px.imshow(
        pivot,
        aspect="auto",
        labels=dict(x="Hour", y="Day", color=metric_key),
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

if st.sidebar.button("Reload", use_container_width=True):
    fetch_csv_text.clear()
    load_wide_df.clear()
    build_long.clear()
    st.experimental_rerun()

# Load & prepare
try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.stop()

tz = "Asia/Bangkok"
now_ts = pd.Timestamp.now(tz=tz)

# Sidebar date range (default 3 days)
min_ts = df_long["timestamp"].min()
max_ts = df_long["timestamp"].max()
default_start = (max_ts - pd.Timedelta(days=2)).date() if pd.notna(max_ts) else pd.Timestamp.now().date()
date1, date2 = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(default_start, max_ts.date() if pd.notna(max_ts) else default_start),
    min_value=min_ts.date() if pd.notna(min_ts) else default_start,
    max_value=max_ts.date() if pd.notna(max_ts) else default_start,
)

if isinstance(date1,(tuple,list)): date1, date2 = date1
start_ts = pd.Timestamp.combine(date1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(date2, pd.Timestamp("23:59").time()).tz_localize(tz)

# Channel filter
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)",
                                options=["[All]"]+all_channels,
                                default=["[All]"])
if ("[All]" in chosen) or not any(c in all_channels for c in chosen):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# Filter timeframe/channels then build hourly
mask = (df_long["timestamp"]>=start_ts) & (df_long["timestamp"]<=end_ts) & (df_long["channel"].isin(selected_channels))
df_sel = df_long.loc[mask].copy()
hourly_all = build_hourly(df_sel)

# Title
st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    if hourly_all.empty:
        st.warning("No data in selected period.")
        st.stop()

    # KPI ณ ชั่วโมงล่าสุด และเทียบเมื่อวานชั่วโมงเดียวกัน
    hmax = hourly_all["hour_key"].max()
    # เอา snapshot ของ 'สะสม' ณ ชั่วโมงนั้น (ฉบับเดิม) : ต้อง map ย้อนกลับหา df_sel
    df_sel["hour_key"] = df_sel["timestamp"].dt.floor("H")

    cur_snap = pick_snapshot_at(df_sel, hmax)
    y_snap   = pick_snapshot_at(df_sel, hmax - pd.Timedelta(days=1))
    cur = kpis_from_snapshot(cur_snap)
    prv = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",
                delta=(f"{pct_delta(cur['Sales'], prv['Sales']):+.1f}%" if pct_delta(cur['Sales'], prv['Sales']) is not None else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}",
                delta=(f"{pct_delta(cur['Orders'], prv['Orders']):+.1f}%" if pct_delta(cur['Orders'], prv['Orders']) is not None else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                delta=(f"{pct_delta(cur['Ads'], prv['Ads']):+.1f}%" if pct_delta(cur['Ads'], prv['Ads']) is not None else None))
    C[3].metric("sale_ro (Sales/Ads)",
                "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prv['SaleRO']):+.1f}%" if pct_delta(cur['SaleRO'], prv['SaleRO']) is not None else None))
    C[4].metric("ads_ro (avg>0)",
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=None)
    st.caption(f"Snapshot hour: {hmax}")

    metric_key = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)",
                              options=["sales","orders","ads","sale_ro","ads_ro"], index=0)

    build_overlay_by_day(hourly_all, metric_key, title="Trend overlay by day")
    st.markdown("### Prime hours heatmap")
    build_prime_heatmap(hourly_all, metric_key)

# -----------------------------------------------------------------------------
# Channel
# -----------------------------------------------------------------------------
elif page == "Channel":
    if not all_channels:
        st.warning("No channels found.")
        st.stop()
    one = st.selectbox("Pick one channel", options=all_channels, index=0)

    df_one = df_sel[df_sel["channel"]==one].copy()
    hourly_one = build_hourly(df_one)
    if hourly_one.empty:
        st.warning("No data for this channel in selected period.")
        st.stop()

    # KPI
    hmax = hourly_one["hour_key"].max()
    cur_snap = pick_snapshot_at(df_one, hmax)
    y_snap   = pick_snapshot_at(df_one, hmax - pd.Timedelta(days=1))
    cur = kpis_from_snapshot(cur_snap)
    prv = kpis_from_snapshot(y_snap)

    C = st.columns(5)
    C[0].metric("Sales", f"{cur['Sales']:,.0f}",
                delta=(f"{pct_delta(cur['Sales'], prv['Sales']):+.1f}%" if pct_delta(cur['Sales'], prv['Sales']) is not None else None))
    C[1].metric("Orders", f"{cur['Orders']:,.0f}",
                delta=(f"{pct_delta(cur['Orders'], prv['Orders']):+.1f}%" if pct_delta(cur['Orders'], prv['Orders']) is not None else None))
    C[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                delta=(f"{pct_delta(cur['Ads'], prv['Ads']):+.1f}%" if pct_delta(cur['Ads'], prv['Ads']) is not None else None))
    C[3].metric("sale_ro (Sales/Ads)",
                "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                delta=(f"{pct_delta(cur['SaleRO'], prv['SaleRO']):+.1f}%" if pct_delta(cur['SaleRO'], prv['SaleRO']) is not None else None))
    C[4].metric("ads_ro (avg>0)",
                "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                delta=None)
    st.caption(f"Snapshot hour: {hmax}")

    metric_key = st.selectbox("Metric to plot (channel):",
                              options=["sales","orders","ads","sale_ro","ads_ro"], index=0)

    build_overlay_by_day(hourly_one, metric_key, title="Trend overlay by day (channel)")

    # Table – ไม่ใช้ melt
    st.markdown("### Time series table (hourly)")
    show_cols = ["hour_key","day","hour",
                 "sales_hour","orders_hour","ads_hour",
                 "sale_ro_hour","ads_ro_hour"]
    tbl = hourly_one[show_cols].sort_values("hour_key").copy()
    tbl = tbl.rename(columns={
        "hour_key":"hour_ts",
        "sales_hour":"sales",
        "orders_hour":"orders",
        "ads_hour":"ads",
        "sale_ro_hour":"sale_ro",
        "ads_ro_hour":"ads_ro"
    })
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=360)

# -----------------------------------------------------------------------------
# Compare (เบา ๆ)
# -----------------------------------------------------------------------------
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels,
                          default=all_channels[:2] if len(all_channels)>=2 else all_channels,
                          max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()
    sub = df_sel[df_sel["channel"].isin(pick)].copy()
    hourly_sub = build_hourly(sub)
    if hourly_sub.empty:
        st.warning("No data for selected channels.")
        st.stop()

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met  = st.selectbox("Metric", options=["sales","orders","ads","sale_ro","ads_ro"], index=0)
    use_col = METRIC_MAP[met]

    piv = hourly_sub.pivot_table(index="hour_key", columns="channel", values=use_col, aggfunc="sum").sort_index()
    piv = piv.fillna(0)
    if base not in piv.columns:
        st.info("Baseline not in data."); st.stop()

    # สัมพัทธ์ต่อฐาน (ถ้าฐานเป็น 0 ป้องกันหารศูนย์)
    base_vals = piv[base].replace(0, np.nan)
    rel = (piv.div(base_vals, axis=0) - 1.0) * 100.0
    rel = rel.drop(columns=[base], errors="ignore")

    fig = go.Figure()
    for c in rel.columns:
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (metric level)")
    sm = hourly_sub.pivot_table(index="hour_key", columns="channel", values=use_col, aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)
