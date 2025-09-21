# app.py
import os, io, re, unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Shopee ROAS", layout="wide")

# -----------------------------
# Utilities: clean header & detect time columns
# -----------------------------
def clean_header(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    for bad in ["\xa0", "\u00A0", "\u202F", "\u2007", "\u2009", "\ufeff", "\u200b", "\u200c", "\u200d"]:
        s = s.replace(bad, " ")
    s = "".join(ch for ch in s if ch.isalnum() or ch in " :")
    s = re.sub(r"\s+", " ", s).strip()
    return s

TIME_COL_PATTERN = re.compile(r"^[A-Za-z]\s*\d{1,2}\s+\d{1,2}:\d{1,2}$")

def is_time_col(col: str) -> bool:
    return TIME_COL_PATTERN.match(clean_header(col)) is not None

def parse_timestamp_from_header(hdr: str, tz="Asia/Bangkok") -> pd.Timestamp:
    h = clean_header(hdr)
    m = re.match(r"^[A-Za-z]\s*(\d{1,2})\s+(\d{1,2}):(\d{1,2})$", h)
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    try:
        return pd.Timestamp(year=now.year, month=now.month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

# -----------------------------
# Parse metric cell -> numbers
# -----------------------------
def str_to_metrics_tuple(s: str):
    # keep first 6: budget,user,order,view,sale,ro
    if not isinstance(s, str):
        return (np.nan,)*6
    if not re.search(r"\d", s):
        return (np.nan,)*6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p!=""]
    nums = []
    for p in parts[:6]:
        try: nums.append(float(p))
        except: nums.append(np.nan)
    while len(nums) < 6: nums.append(np.nan)
    return tuple(nums[:6])

# -----------------------------
# Wide -> Long
# -----------------------------
def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    df_wide = df_wide.rename(columns=lambda c: clean_header(c))
    id_cols, time_cols = [], []
    for c in df_wide.columns:
        if clean_header(c).lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")
    df_melt["timestamp"] = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))

    metrics = df_melt["metrics"].apply(str_to_metrics_tuple)
    metrics_df = pd.DataFrame(metrics.tolist(),
                              columns=["budget","user","order","view","sale","ro"])
    out = pd.concat([df_melt[["timestamp"] + id_cols], metrics_df], axis=1)
    out.rename(columns={"name":"channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    # KPIs
    out["ROAS"] = out["sale"]   / out["budget"].replace(0, np.nan)
    out["AOV"]  = out["sale"]   / out["order"].replace(0, np.nan)
    out["CPO"]  = out["budget"] / out["order"].replace(0, np.nan)
    out["RPV"]  = out["sale"]   / out["view"].replace(0, np.nan)
    out["ORV"]  = out["order"]  / out["view"].replace(0, np.nan)
    return out

# -----------------------------
# Data loading (from Secret)
# -----------------------------
@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_sheet():
    url = os.environ.get("ROAS_CSV_URL", "").strip()
    if not url:
        raise RuntimeError("Missing secret ROAS_CSV_URL")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.rename(columns=lambda c: clean_header(c), inplace=True)
    return df

@st.cache_data(ttl=60*10, show_spinner=False)
def build_long(df):
    return long_from_wide(df)

# -----------------------------
# Helpers
# -----------------------------
def latest_per_hour(df):
    x = df.copy()
    x["hour"] = x["timestamp"].dt.floor("H")
    idx = x.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return x.loc[idx]

def compute_totals(df):
    if df is None or df.empty:
        return dict(Sales=np.nan, Orders=np.nan, Budget=np.nan,
                    ROAS=np.nan, AOV=np.nan, CPO=np.nan, RPV=np.nan, ORV=np.nan)
    tot = {}
    tot["Sales"]  = df["sale"].sum()
    tot["Orders"] = df["order"].sum()
    tot["Budget"] = df["budget"].sum()
    tot["ROAS"]   = tot["Sales"]/tot["Budget"] if tot["Budget"] else np.nan
    tot["AOV"]    = tot["Sales"]/tot["Orders"] if tot["Orders"] else np.nan
    tot["CPO"]    = tot["Budget"]/tot["Orders"] if tot["Orders"] else np.nan
    tot["RPV"]    = df["sale"].sum()/df["view"].sum() if df["view"].sum() else np.nan
    tot["ORV"]    = df["order"].sum()/df["view"].sum() if df["view"].sum() else np.nan
    return tot

def pct_delta(curr, prev):
    if prev in [0, None] or pd.isna(prev): return None
    if curr is None or pd.isna(curr): return None
    return (curr - prev) * 100.0 / prev

def kpi_row(tot, base_tot):
    labels = ["Sales","Orders","Budget","ROAS","AOV","CPO","RPV","ORV"]
    cols = st.columns(8)
    for i, lab in enumerate(labels):
        curr = tot.get(lab, np.nan)
        prev = base_tot.get(lab, np.nan)
        delta = pct_delta(curr, prev) if prev is not None else None
        if delta is None:
            cols[i].metric(lab, f"{curr:,.0f}" if pd.notna(curr) else "-")
        else:
            cols[i].metric(lab, f"{curr:,.0f}" if pd.notna(curr) else "-", f"{delta:+.1f}%")

# -----------------------------
# Header area
# -----------------------------
st.title("Shopee ROAS Dashboard")
btn_col, _ = st.columns([1,3])
with btn_col:
    if st.button("Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
st.caption("Last refresh: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# Load data
try:
    df_wide = fetch_sheet()
    df_long = build_long(df_wide)
except Exception as e:
    st.error(f"โหลด/แปลงข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# -----------------------------
# Sidebar: Filters + Page selector
# -----------------------------
st.sidebar.header("Filters")

channels_all = sorted(df_long["channel"].dropna().unique().tolist())
tz = "Asia/Bangkok"
max_ts = df_long["timestamp"].max()
min_ts = df_long["timestamp"].min()

# Default 3 days
start_def = (max_ts - pd.Timedelta(days=3-1)).date()
end_def   = max_ts.date()
d1, d2 = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(start_def, end_def),
    min_value=min_ts.date(), max_value=max_ts.date()
)
if isinstance(d1, (list, tuple)):
    d1, d2 = d1

channel_filter = st.sidebar.multiselect(
    "Channels (เลือกรวม All ได้)",
    options=["[All]"] + channels_all,
    default=["[All]"]
)

page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])

# Build masks (date first)
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)
date_mask = df_long["timestamp"].between(start_ts, end_ts)

# Channel mask for Overview only (Channel/Compare จะเลือกจากทุกช่องในชีต)
if "[All]" in channel_filter or len(channel_filter)==0:
    ch_mask_overview = df_long["channel"].isin(channels_all)
else:
    ch_mask_overview = df_long["channel"].isin(channel_filter)

# Baseline (yesterday same range)
y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts   - pd.Timedelta(days=1)
base_date_mask = df_long["timestamp"].between(y_start, y_end)

# -----------------------------
# PAGES
# -----------------------------
if page == "Overview":
    d = df_long.loc[date_mask & ch_mask_overview].copy()
    if d.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    baseline = df_long.loc[base_date_mask & ch_mask_overview].copy()
    hourly = latest_per_hour(d)
    base_hourly = latest_per_hour(baseline) if not baseline.empty else None

    tot = compute_totals(hourly)
    base_tot = compute_totals(base_hourly) if base_hourly is not None else {}

    kpi_row(tot, base_tot)

    st.markdown("### Trend by hour (Sales/Orders/ROAS)")
    trend = hourly.groupby("hour").agg({"sale":"sum","order":"sum","budget":"sum"}).reset_index()
    trend["ROAS"] = trend["sale"] / trend["budget"].replace(0, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["sale"],  name="Sales",  mode="lines+markers"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["order"], name="Orders", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=trend["hour"], y=trend["ROAS"],  name="ROAS",  mode="lines+markers", yaxis="y3"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="ROAS",  overlaying="y", side="right", position=1.0),
        legend=dict(orientation="h", y=-0.2),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prime hours heatmap")
    tmp = hourly.copy()
    tmp["day"] = tmp["hour"].dt.date
    tmp["hr"]  = tmp["hour"].dt.hour
    heat = tmp.groupby(["day","hr"]).agg(val=("sale","sum")).reset_index()
    heat_pivot = heat.pivot(index="day", columns="hr", values="val").sort_index(ascending=False)
    fig_h = px.imshow(heat_pivot, aspect="auto", color_continuous_scale="YlOrRd",
                      labels=dict(x="Hour", y="Day", color="Sales"))
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("### Data (hourly latest snapshot per channel)")
    show_cols = ["hour","channel","budget","user","order","view","sale","ROAS","AOV","CPO","RPV","ORV"]
    st.dataframe(hourly[show_cols].sort_values(["hour","channel"]).round(3), use_container_width=True)

elif page == "Channel":
    # เลือกได้จาก "ทุกช่องในชีต" ไม่ขึ้นกับ Filters
    pick_one = st.selectbox("Pick one channel (all channels in sheet)", options=channels_all)
    d = df_long.loc[date_mask & (df_long["channel"]==pick_one)].copy()
    baseline = df_long.loc[base_date_mask & (df_long["channel"]==pick_one)].copy()

    hourly = latest_per_hour(d)
    base_hourly = latest_per_hour(baseline) if not baseline.empty else None
    tot = compute_totals(hourly)
    base_tot = compute_totals(base_hourly) if base_hourly is not None else {}
    st.subheader(f"Channel • {pick_one}")
    kpi_row(tot, base_tot)

    st.markdown("### Multi-axis line")
    tr = hourly.sort_values("hour")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["sale"],   name="Sales",  mode="lines+markers"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["order"],  name="Orders", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["budget"], name="Budget", mode="lines+markers", yaxis="y3"))
    fig.add_trace(go.Scatter(x=tr["hour"], y=tr["ROAS"],   name="ROAS",  mode="lines+markers", yaxis="y4"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        yaxis3=dict(title="Budget", overlaying="y", side="right", position=1.0),
        yaxis4=dict(title="ROAS",   overlaying="y", side="right", position=0.95),
        legend=dict(orientation="h", y=-0.3),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 24h distribution")
    tr["h"] = tr["hour"].dt.hour
    dbar = tr.groupby("h").agg(Sales=("sale","sum"), ORV=("ORV","mean"), RPV=("RPV","mean")).reset_index()
    st.plotly_chart(px.bar(dbar, x="h", y="Sales"), use_container_width=True)
    st.caption("Bars = Sales/hour; hover เพื่อดู ORV/RPV (mean)")

    st.markdown("### Time series table")
    st.dataframe(tr[["hour","budget","user","order","view","sale","ROAS","AOV","CPO","RPV","ORV"]].round(3),
                 use_container_width=True)

else:  # Compare
    pick_multi = st.multiselect("Pick 2–4 channels (all channels in sheet)",
                                options=channels_all, default=channels_all[:2])
    if len(pick_multi) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()
    if len(pick_multi) > 4:
        pick_multi = pick_multi[:4]

    sub = df_long.loc[date_mask & (df_long["channel"].isin(pick_multi))].copy()
    hourly = latest_per_hour(sub)

    st.subheader(f"Compare: {', '.join(pick_multi)}")
    kpis = hourly.groupby("channel").agg(
        ROAS=("ROAS","mean"), AOV=("AOV","mean"), CPO=("CPO","mean"),
        RPV=("RPV","mean"), ORV=("ORV","mean")
    ).reset_index()
    st.markdown("### KPI comparison table")
    st.dataframe(kpis.round(3), use_container_width=True)

    base = st.selectbox("Baseline channel", options=pick_multi, index=0)
    met = st.selectbox("Metric", options=["ROAS","sale","order","budget"], index=0)

    pivot = hourly.pivot_table(index="hour", columns="channel", values=met, aggfunc="sum").sort_index()
    rel = (pivot.div(pivot[base], axis=0)-1.0)*100.0
    fig = go.Figure()
    for c in rel.columns:
        if c == base: 
            continue
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], mode="lines+markers", name=f"{c} vs {base}"))
    fig.update_layout(yaxis_title="% difference", xaxis_title="Hour",
                      height=420, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Small multiples (ROAS)")
    sm = hourly.pivot_table(index="hour", columns="channel", values="ROAS", aggfunc="mean").sort_index()
    fig2 = go.Figure()
    for c in sm.columns:
        fig2.add_trace(go.Scatter(x=sm.index, y=sm[c], mode="lines", name=c))
    fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig2, use_container_width=True)
