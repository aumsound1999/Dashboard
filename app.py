# app.py
# Shopee ROAS Dashboard — Overview • Channel • Compare
# อ่านข้อมูลจาก Google Sheet (CSV export) ผ่าน Secrets:
#   ROAS_CSV_URL="https://docs.google.com/spreadsheets/d/<ID>/gviz/tqx=out:csv&sheet=<SHEET>"
#
# pip: streamlit pandas numpy plotly requests

import os, io, re
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers: detect & parse
# -----------------------------------------------------------------------------

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamp_from_header(hdr: str, tz: str = "Asia/Bangkok") -> pd.Timestamp:
    """
    "D21 12:45" -> day=21 hour=12 minute=45 ; ใช้ year/month ปัจจุบัน
    """
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
    """
    รับค่าเป็น string ตัวเลขคั่นด้วย comma -> list ความยาว 6 (เติม NaN ถ้าไม่ครบ)
    ตัวอย่าง "2025,12,34,776,22.51,1036"
    mapping:
      v0=sales, v1=orders, v2=ads, v3=view, v4=ads_ro(raw), v5=misc
    """
    if not isinstance(s, str) or not re.search(r"\d", s or ""):
        return [np.nan]*6
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
# Loaders
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_csv_text():
    url = os.environ.get("ROAS_CSV_URL", "")
    if not url:
        raise RuntimeError("Missing Secrets: ROAS_CSV_URL")
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=600, show_spinner=True)
def load_wide_df():
    return pd.read_csv(io.StringIO(fetch_csv_text()))

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # แยก id/time columns
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(str(c)):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:45'.")

    m = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                     var_name="time_col", value_name="raw")
    m["timestamp"] = m["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))

    V = pd.DataFrame(m["raw"].apply(parse_metrics_cell).tolist(),
                     columns=["v0", "v1", "v2", "v3", "v4", "v5"])

    out = pd.concat([m[["timestamp"] + id_cols], V], axis=1).rename(columns={"name": "channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # named metrics
    out["sales"]   = pd.to_numeric(out["v0"], errors="coerce")
    out["orders"]  = pd.to_numeric(out["v1"], errors="coerce")
    out["ads"]     = pd.to_numeric(out["v2"], errors="coerce")
    out["view"]    = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro_raw"] = pd.to_numeric(out["v4"], errors="coerce")  # ไม่ใช้ตรงๆในกราฟ
    out["misc"]    = pd.to_numeric(out["v5"], errors="coerce")

    # ROAS รวมแบบสะสม ณ เวลา (ใช้สำหรับ KPI หัวตารางเท่านั้น)
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)

    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(wide):
    return long_from_wide(wide)

# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------

def safe_tz(ts: pd.Series, tz="Asia/Bangkok"):
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)
    except Exception:
        pass
    return ts

def normalize_hour_key(df: pd.DataFrame, tz="Asia/Bangkok"):
    """มีเมื่อไหร่ก็สร้าง hour_key ให้เสมอ"""
    if "hour_key" not in df.columns:
        if "timestamp" in df.columns:
            hk = safe_tz(df["timestamp"], tz=tz).dt.floor("H")
        elif {"day", "hour"}.issubset(df.columns):
            day = pd.to_datetime(df["day"], errors="coerce")
            hour = pd.to_timedelta(pd.to_numeric(df["hour"], errors="coerce").fillna(0), unit="h")
            hk = (day + hour)
            try:
                hk = hk.dt.tz_localize(tz)
            except Exception:
                pass
        else:
            hk = pd.NaT
        df = df.copy()
        df["hour_key"] = hk
    return df

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp, tz: str = "Asia/Bangkok") -> pd.DataFrame:
    """เลือกแถวล่าสุดของแต่ละ channel ที่อยู่ในชั่วโมงเดียวกับ at_ts (floor ชั่วโมง)
       กัน KeyError กรณีไม่มี hour_key
    """
    if df is None or df.empty:
        return df
    x = normalize_hour_key(df, tz=tz)

    if isinstance(at_ts, pd.Timestamp):
        try:
            at_ts = at_ts.tz_localize(tz) if at_ts.tz is None else at_ts.tz_convert(tz)
        except Exception:
            pass
    target = at_ts.floor("H")

    y = x[x["hour_key"] == target].copy()
    if y.empty: 
        return y
    sort_key = "timestamp" if "timestamp" in y.columns else "hour_key"
    y = y.sort_values(sort_key)
    if "channel" in y.columns:
        y = y.groupby("channel").tail(1).reset_index(drop=True)
    else:
        y = y.tail(1).reset_index(drop=True)
    return y

def current_and_yesterday_snapshots(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty:
        return df, df, pd.NaT
    cur_ts = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts, tz=tz)
    y_snap = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1), tz=tz)
    return cur_snap, y_snap, cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales"].sum()
    orders = snap["orders"].sum()
    ads = snap["ads"].sum()
    sale_ro = (sales / ads) if ads != 0 else np.nan
    ads_ro_vals = snap["ads_ro_raw"]
    ads_ro_avg = ads_ro_vals[ads_ro_vals > 0].mean()
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev) or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

# --- Hourly latest snapshot per hour (ไม่เชื่อมต่อกันข้าม snapshot)
def hourly_latest(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty: 
        return df.copy()
    d = df.copy()
    d["hour_key"] = safe_tz(d["timestamp"], tz=tz).dt.floor("H")
    d = d.sort_values("timestamp").groupby(["channel", "hour_key"]).tail(1)
    d["day"]  = d["hour_key"].dt.date
    d["hstr"] = d["hour_key"].dt.strftime("%H:%M")
    return d.reset_index(drop=True)

# --- คำนวณค่า overlay/heatmap ตามสเปค: diff สำหรับ sales/orders/ads, ro เป็นรายชั่วโมง capped 50
def build_overlay_by_day(df: pd.DataFrame, metric: str, tz="Asia/Bangkok"):
    """คืน DataFrame index=hour (string HH:MM), columns=day; values = ตามกติกา"""
    if df.empty:
        return pd.DataFrame()

    H = hourly_latest(df, tz=tz).sort_values(["channel", "hour_key"])
    # เตรียมคอลัมน์ที่เอาไปคำนวณ
    # difference ป้องกันค่าติดลบ (rollover/รีเซ็ต/เขียนซ้ำ) -> clip lower=0
    def per_channel_hourly_diff(s):
        diff = s.diff()
        diff = diff.clip(lower=0)  # กันติดลบ
        return diff

    if metric in ("sales", "orders", "ads"):
        diff_col = H.groupby("channel")[metric].apply(per_channel_hourly_diff).reset_index(level=0, drop=True)
        H["_val"] = diff_col.fillna(0.0)
    elif metric in ("sale_ro", "ads_ro"):
        # สร้างยอดรายชั่วโมงจาก diff ของ sales/ads
        ds = H.groupby("channel")["sales"].apply(per_channel_hourly_diff).reset_index(level=0, drop=True).fillna(0.0)
        da = H.groupby("channel")["ads"].apply(per_channel_hourly_diff).reset_index(level=0, drop=True).replace(0, np.nan)
        ro = ds / da
        ro = ro.replace([np.inf, -np.inf], np.nan).clip(upper=50)  # cap 50
        H["_val"] = ro.fillna(0.0)
    else:
        # ค่าอื่นไม่รองรับก็ fallback เป็น 0
        H["_val"] = 0.0

    pivot = H.pivot_table(index="hstr", columns="day", values="_val", aggfunc="sum").sort_index()
    return pivot

# -----------------------------------------------------------------------------
# UI: Sidebar
# -----------------------------------------------------------------------------

st.sidebar.header("Filters")

# ปุ่มรีโหลด cache
col_reload = st.sidebar.columns([1,3,3])[0]
with col_reload:
    if st.button("Reload", use_container_width=True):
        fetch_csv_text.clear(); load_wide_df.clear(); build_long.clear()
        st.experimental_rerun()

# โหลดข้อมูล
try:
    wide = load_wide_df()
    df_long = build_long(wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.stop()

tz = "Asia/Bangkok"
now_ts = pd.Timestamp.now(tz=tz)

# date range
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

# Channels [All]
all_channels = sorted(df_long["channel"].dropna().unique().tolist())
chan_options = ["[All]"] + all_channels
chosen = st.sidebar.multiselect("Channels (เลือก All ได้)", options=chan_options, default=["[All]"])
if ("[All]" in chosen) or (not any(c in all_channels for c in chosen)):
    selected_channels = all_channels
else:
    selected_channels = [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])
st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# กรองช่วง + ช่อง
mask = (
    (df_long["timestamp"] >= start_ts) &
    (df_long["timestamp"] <= end_ts) &
    (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

# ==== Overview ====
if page == "Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d, tz=tz)
    cur = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)

    cols = st.columns(5)
    cols[0].metric("Sales", f"{cur['Sales']:,.0f}",
                   delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    cols[1].metric("Orders", f"{cur['Orders']:,.0f}",
                   delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    cols[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                   delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    cols[3].metric("sale_ro (Sales/Ads)",
                   "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                   delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    cols[4].metric("ads_ro (avg>0)",
                   "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                   delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # Trend overlay by day (เลือกได้ 1 metric)
    st.markdown("### Trend overlay by day")
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", options=["sales", "orders", "ads", "sale_ro", "ads_ro"], index=0)

    piv = build_overlay_by_day(d, metric, tz=tz)
    if piv.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for day in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[day], mode="lines+markers", name=str(day)))
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    # Prime hours heatmap (ใช้กติกาเดียวกับ overlay)
    st.markdown("### Prime hours heatmap")
    piv_hm = piv.copy()
    if not piv_hm.empty:
        fig_hm = px.imshow(
            piv_hm.T,  # day x hour
            aspect="auto",
            labels=dict(x="Hour", y="Day", color=metric),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No data for heatmap.")

    # Raw hourly latest table (ปลอด error matplotlib)
    st.markdown("### Data (hourly latest snapshot per channel)")
    show = hourly_latest(d, tz=tz)[["hour_key", "channel", "ads", "orders", "sales", "SaleRO", "ads_ro_raw"]]
    show = show.rename(columns={"hour_key": "hour", "ads": "budget(ads)", "ads_ro_raw": "ads_ro"})
    st.dataframe(show.sort_values(["hour", "channel"]).round(3), use_container_width=True, height=360)

# ==== Channel ====
elif page == "Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = df_long[
        (df_long["channel"] == ch) &
        (df_long["timestamp"] >= start_ts) &
        (df_long["timestamp"] <= end_ts)
    ].copy()
    if ch_df.empty:
        st.warning("No data for this channel in selected period.")
        st.stop()

    # KPIs
    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df, tz=tz)
    cur = kpis_from_snapshot(cur_snap)
    prev = kpis_from_snapshot(y_snap)
    cols = st.columns(5)
    cols[0].metric("Sales", f"{cur['Sales']:,.0f}",
                   delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    cols[1].metric("Orders", f"{cur['Orders']:,.0f}",
                   delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    cols[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                   delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    cols[3].metric("sale_ro (Sales/Ads)",
                   "-" if pd.isna(cur["SaleRO"]) else f"{cur['SaleRO']:.3f}",
                   delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    cols[4].metric("ads_ro (avg>0)",
                   "-" if pd.isna(cur["AdsRO_avg"]) else f"{cur['AdsRO_avg']:.2f}",
                   delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    st.caption(f"Snapshot hour: {cur_hour}")

    # Trend overlay (เฉพาะ channel นี้)
    st.markdown("### Trend overlay by day (channel)")
    metric = st.selectbox("Metric to plot (channel):", options=["sales", "orders", "ads", "sale_ro", "ads_ro"], index=0)
    piv = build_overlay_by_day(ch_df, metric, tz=tz)
    if piv.empty:
        st.info("No data to plot.")
    else:
        fig = go.Figure()
        for day in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[day], mode="lines+markers", name=str(day)))
        fig.update_layout(height=420, xaxis_title="Hour", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)

    # ตารางต่อชั่วโมง
    st.markdown("### Time series table (hourly)")
    H = hourly_latest(ch_df, tz=tz).sort_values("hour_key")
    # เติมค่ารายชั่วโมงตามตรรกะเดียวกัน เพื่อดูเทียบกับกราฟ
    if metric in ("sales", "orders", "ads"):
        series = H.groupby("channel")[metric].diff().clip(lower=0).fillna(0.0)
    else:
        ds = H.groupby("channel")["sales"].diff().clip(lower=0).fillna(0.0)
        da = H.groupby("channel")["ads"].diff().clip(lower=0).replace(0, np.nan)
        series = (ds/da).replace([np.inf, -np.inf], np.nan).clip(upper=50).fillna(0.0)
    out = H[["hour_key", "channel", "ads", "orders", "sales", "SaleRO", "ads_ro_raw"]].copy()
    out["metric_value"] = series.values
    out = out.rename(columns={"hour_key":"hour", "ads": "budget(ads)", "ads_ro_raw":"ads_ro"})
    st.dataframe(out.round(3), use_container_width=True, height=380)

# ==== Compare ====
else:
    pick = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:2], max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()

    sub = df_long[
        (df_long["channel"].isin(pick)) &
        (df_long["timestamp"] >= start_ts) &
        (df_long["timestamp"] <= end_ts)
    ].copy()
    if sub.empty:
        st.warning("No data for selected channels in range.")
        st.stop()

    # สแนปช็อตล่าสุดต่อชั่วโมง/ช่อง (คงที่)
    H = hourly_latest(sub, tz=tz)

    st.subheader(f"Compare: {', '.join(pick)}")
    # KPI table (ง่าย ๆ)
    kpis = H.groupby("channel").agg(
        ROAS=("SaleRO", "mean"),
        AOV=("sales", lambda s: (s.sum() / H.loc[s.index, "orders"].sum()) if H.loc[s.index, "orders"].sum() else np.nan),
        CPO=("orders", lambda s: (H.loc[s.index, "ads"].sum() / s.sum()) if s.sum() else np.nan),
        RPV=("sales", lambda s: (s.sum() / H.loc[s.index, "view"].sum()) if H.loc[s.index, "view"].sum() else np.nan),
        ORV=("orders", lambda s: (s.sum() / H.loc[s.index, "view"].sum()) if H.loc[s.index, "view"].sum() else np.nan),
    ).reset_index()
    st.markdown("#### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met = st.selectbox("Metric", options=["ROAS", "sales", "orders", "ads"], index=0)
    piv = H.pivot_table(index="hour_key", columns="channel", values=("SaleRO" if met=="ROAS" else met), aggfunc="sum").sort_index()
    rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0

    fig = go.Figure()
    for c in rel.columns:
        if c == base: 
            continue
        fig.add_trace(go.Scatter(x=rel.index.strftime("%H:%M"), y=rel[c], name=f"{c} vs {base}", mode="lines+markers"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="% difference")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Small multiples (ROAS)")
    sm = H.pivot_table(index="hour_key", columns="channel", values="SaleRO", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index.strftime("%H:%M"), y=sm[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig2, use_container_width=True)
