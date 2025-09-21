# app.py
# Streamlit dashboard: Overview • Channel • Compare
# - KPI: ใช้ cumulative snapshot ล่าสุด
# - กราฟ/ฮีตแมป: ใช้ค่า increment รายชั่วโมง (diff ในวันเดียวกัน) และ clip ค่าติดลบเป็น 0
# - RO (sale_ro_i / ads_ro_i) capped ที่ 50
# - ถ้ามีคอลัมน์ sale_from_ads (ยอดขายจากแอดแบบสะสม) ให้ map ใส่เพื่อคำนวณ ads_ro_i ตามจริง

import os, re, io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------------------------------------------------------
# Helpers
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
    """แปลงสตริงคอมมา -> list ความยาว 6"""
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
        raise ValueError("No time columns detected. Expect headers like 'D21 12:45'.")

    melted = df_wide.melt(
        id_vars=id_cols, value_vars=time_cols,
        var_name="time_col", value_name="raw"
    )
    melted["timestamp"] = melted["time_col"].apply(lambda x: parse_timestamp_from_header(str(x), tz=tz))
    parsed = melted["raw"].apply(parse_metrics_cell)
    V = pd.DataFrame(parsed.tolist(), columns=["v0","v1","v2","v3","v4","v5"])

    out = pd.concat([melted[["timestamp"] + id_cols], V], axis=1).rename(columns={"name": "channel"})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Mapping คอลัมน์จากชีต
    out["sales"]      = pd.to_numeric(out["v0"], errors="coerce")  # cumulative
    out["orders"]     = pd.to_numeric(out["v1"], errors="coerce")
    out["ads"]        = pd.to_numeric(out["v2"], errors="coerce")  # ad spend cumulative
    out["views"]      = pd.to_numeric(out["v3"], errors="coerce")
    out["ads_ro_raw"] = pd.to_numeric(out["v4"], errors="coerce")  # optional RO per channel
    out["misc"]       = pd.to_numeric(out["v5"], errors="coerce")

    # ถ้ามีคอลัมน์ยอดขายจากแอดสะสม ให้ map เพิ่ม (ตัวอย่าง)
    # out["sale_from_ads"] = pd.to_numeric(out["v6"], errors="coerce")

    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(df_wide):
    return long_from_wide(df_wide)

def add_day_hour_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day"]  = out["timestamp"].dt.date
    out["hhmm"] = out["timestamp"].dt.strftime("%H:%M")
    return out

def build_hourly_increments(df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลง cumulative -> increment รายชั่วโมง (ภายใน channel/day)
    ตัดค่าติดลบเป็น 0 และคำนวณ RO แบบรายชั่วโมง (cap 50)
    """
    sale_from_ads_col = "sale_from_ads" if "sale_from_ads" in df.columns else None

    need = ["channel","timestamp","day","hhmm","sales","orders","ads"]
    if sale_from_ads_col:
        need.append(sale_from_ads_col)

    d = df[need].sort_values(["channel","day","timestamp"]).copy()
    g = d.groupby(["channel","day"], group_keys=False)

    def _inc(col):
        return g[col].diff().clip(lower=0)

    d["sales_i"]  = _inc("sales")
    d["orders_i"] = _inc("orders")
    d["ads_i"]    = _inc("ads")

    eps = 1e-9
    d["sale_ro_i"] = (d["sales_i"] / (d["ads_i"] + eps)).clip(upper=50)

    if sale_from_ads_col:
        d["sale_ads_i"] = _inc(sale_from_ads_col)
        d["ads_ro_i"]   = (d["sale_ads_i"] / (d["ads_i"] + eps)).clip(upper=50)
    else:
        # หากไม่มียอดขายจากแอด ใช้ sale_ro_i เป็น proxy
        d["ads_ro_i"] = d["sale_ro_i"]

    return d

def build_overlay_by_day(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """เตรียมข้อมูลกราฟเส้นซ้อนรายวันจากยอดรายชั่วโมง"""
    metric_map = {
        "sales":"sales_i", "orders":"orders_i", "ads":"ads_i",
        "sale_ro":"sale_ro_i", "ads_ro":"ads_ro_i"
    }
    use_col = metric_map[metric]

    df2 = add_day_hour_cols(df)
    inc  = build_hourly_increments(df2)

    # ปริมาณใช้ sum, RO ใช้ mean
    aggfunc = "sum" if use_col in ["sales_i","orders_i","ads_i"] else "mean"

    # **แก้ error**: aggregate บน Series แล้ว rename
    out = (
        inc.groupby(["day","hhmm"], as_index=False)[use_col]
           .agg(aggfunc)
           .rename(columns={use_col:"val"})
           .sort_values(["day","hhmm"])
    )
    return out

def build_heatmap(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric_map = {
        "sales":"sales_i", "orders":"orders_i", "ads":"ads_i",
        "sale_ro":"sale_ro_i", "ads_ro":"ads_ro_i"
    }
    use_col = metric_map[metric]

    df2 = add_day_hour_cols(df)
    inc  = build_hourly_increments(df2)
    vals = inc.groupby(["day","hhmm"], as_index=False)[use_col].mean()
    pivot = vals.pivot(index="day", columns="hhmm", values=use_col)
    pivot = pivot.reindex(sorted(pivot.columns, key=lambda s: s), axis=1)
    return pivot

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    target_hour = at_ts.floor("H")
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
    cur_ts  = df["timestamp"].max()
    cur     = pick_snapshot_at(df, cur_ts)
    yest    = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1))
    return cur, yest, cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0, Orders=0, Ads=0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = snap["sales"].sum()
    orders= snap["orders"].sum()
    ads   = snap["ads"].sum()
    sale_ro = (sales/ads) if ads else np.nan

    if "ads_ro_raw" in snap.columns:
        r = snap["ads_ro_raw"]
        ads_ro_avg = r[r>0].mean()
    else:
        ads_ro_avg = np.nan
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if prev in [0,None] or pd.isna(prev) or pd.isna(curr):
        return None
    return (curr - prev) * 100.0 / prev

# -----------------------------------------------------------------------------
# Load & filters
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
selected_channels = all_channels if ("[All]" in chosen or not any(c in all_channels for c in chosen)) \
                    else [c for c in chosen if c in all_channels]

page = st.sidebar.radio("Page", ["Overview","Channel","Compare"])

mask = (
    (df_long["timestamp"]>=start_ts) &
    (df_long["timestamp"]<=end_ts) &
    (df_long["channel"].isin(selected_channels))
)
d = df_long.loc[mask].copy()

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------

metrics_list = ["sales","orders","ads","sale_ro","ads_ro"]

if page=="Overview":
    st.subheader("Overview (All selected channels)")
    if d.empty:
        st.warning("No data in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d)
    cur = kpis_from_snapshot(cur_snap)
    prev= kpis_from_snapshot(y_snap)

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
    metric = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", options=metrics_list, index=0, key="ov_metric")
    overlay = build_overlay_by_day(d, metric)

    for_day = overlay.pivot(index="hhmm", columns="day", values="val").sort_index()
    fig = go.Figure()
    for day in for_day.columns:
        fig.add_trace(go.Scatter(x=for_day.index, y=for_day[day], mode="lines+markers", name=str(day)))
    fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prime hours heatmap")
    heat = build_heatmap(d, metric)
    fig_h = px.imshow(heat, aspect="auto", labels=dict(x="Hour", y="Day", color=metric))
    st.plotly_chart(fig_h, use_container_width=True)

elif page=="Channel":
    ch = st.selectbox("Pick one channel", options=all_channels, index=0)
    ch_df = d[d["channel"]==ch].copy()
    if ch_df.empty:
        st.warning("No data for this channel in selected period.")
        st.stop()

    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df)
    cur = kpis_from_snapshot(cur_snap)
    prev= kpis_from_snapshot(y_snap)

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
    metric_c = st.selectbox("Metric to plot (เลือกได้ 1 ค่า)", options=metrics_list, index=0, key="ch_metric")
    overlay = build_overlay_by_day(ch_df, metric_c)
    for_day = overlay.pivot(index="hhmm", columns="day", values="val").sort_index()
    fig = go.Figure()
    for day in for_day.columns:
        fig.add_trace(go.Scatter(x=for_day.index, y=for_day[day], mode="lines+markers", name=str(day)))
    fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=metric_c)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Time series table (hourly increments)")
    inc = build_hourly_increments(add_day_hour_cols(ch_df))
    show = inc[["timestamp","sales_i","orders_i","ads_i","sale_ro_i","ads_ro_i"]].rename(
        columns={"sales_i":"sales","orders_i":"orders","ads_i":"ads",
                 "sale_ro_i":"sale_ro","ads_ro_i":"ads_ro"})
    st.dataframe(show.round(3), use_container_width=True, height=420)

else:  # Compare
    pick = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:2], max_selections=4)
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()
    sub = d[d["channel"].isin(pick)].copy()
    if sub.empty:
        st.warning("No data in selected range.")
        st.stop()

    st.subheader(f"Compare: {', '.join(pick)}")

    base = st.selectbox("Baseline channel", options=pick, index=0)
    met  = st.selectbox("Metric", options=["sales","orders","ads","sale_ro","ads_ro"], index=0)

    inc  = build_hourly_increments(add_day_hour_cols(sub))
    metric_map = {"sales":"sales_i","orders":"orders_i","ads":"ads_i","sale_ro":"sale_ro_i","ads_ro":"ads_ro_i"}
    use_col = metric_map[met]
    piv = inc.pivot_table(index="timestamp", columns="channel", values=use_col, aggfunc="sum").sort_index()

    if base in piv.columns:
        rel = (piv.div(piv[base], axis=0) - 1.0) * 100.0
        fig = go.Figure()
        for c in rel.columns:
            if c == base: continue
            fig.add_trace(go.Scatter(x=rel.index, y=rel[c], mode="lines+markers", name=f"{c} vs {base}"))
        fig.update_layout(height=420, xaxis_title="Time", yaxis_title="% difference")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Small multiples")
    fig2 = go.Figure()
    for c in piv.columns:
        fig2.add_trace(go.Scatter(x=piv.index, y=piv[c], name=c, mode="lines"))
    fig2.update_layout(height=360, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)
