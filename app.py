# app.py
import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# -------------------- Streamlit base --------------------
st.set_page_config(page_title="Shopee ROAS Dashboard", layout="wide")

# -------------------- Helpers (parse wide sheet) --------------------
TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")  # e.g., D21 12:4

def is_time_col(col: str) -> bool:
    s = str(col).strip()
    if not s:
        return False
    return TIME_COL_PATTERN.match(s) is not None

def parse_timestamp_from_header(hdr: str, year=None, month=None, tz="Asia/Bangkok") -> pd.Timestamp:
    # "D21 12:4" -> day=21, hour=12, minute=4
    m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr.strip())
    if not m:
        return pd.NaT
    d, hh, mm = map(int, m.groups())
    now = pd.Timestamp.now(tz=tz)
    if year is None: year = now.year
    if month is None: month = now.month
    try:
        return pd.Timestamp(year=year, month=month, day=d, hour=hh, minute=mm, tz=tz)
    except Exception:
        return pd.NaT

def parse_metrics_c_style(s: str):
    """
    ค่าที่อยู่ในแต่ละ cell ของคอลัมน์เวลา (เช่น 'D21 12:04') เป็น comma numbers
    โดยนิยามใหม่ (ตามที่คุณต้องการใช้เป็นหัว):
        index 0 -> sales
        index 1 -> orders
        index 2 -> ads
        index 4 -> ads_ro   (ถ้ามี ไม่งั้น NaN)
    ส่วน sale_ro จะไปคำนวณตอนรวม (sum(sales)/sum(ads))

    รูปแบบจริงอาจมี 6-7 ตัว ก็จะหยิบตามตำแหน่งที่อธิบายไว้
    """
    if not isinstance(s, str):
        return (np.nan, np.nan, np.nan, np.nan)
    if not re.search(r"\d", s):
        return (np.nan, np.nan, np.nan, np.nan)

    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = [p for p in s_clean.split(",") if p != ""]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except:
            nums.append(np.nan)

    sales = nums[0] if len(nums) >= 1 else np.nan
    orders = nums[1] if len(nums) >= 2 else np.nan
    ads   = nums[2] if len(nums) >= 3 else np.nan
    ads_ro = nums[4] if len(nums) >= 5 else np.nan  # บางกรณีไม่มี index 4
    return (sales, orders, ads, ads_ro)

def wide_to_long(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    """
    แปลง wide sheet -> long:
      - id: 'name'
      - value: time columns (เช่น D21 12:04, D21 13:xx ...)
      - metrics: sales, orders, ads, ads_ro (sale_ro ไปคำนวณตอนรวม)
    """
    cols = list(df_wide.columns)
    id_cols, time_cols = [], []
    for c in cols:
        if str(c).strip().lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
        else:
            pass

    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    melted = df_wide.melt(
        id_vars=id_cols,
        value_vars=time_cols,
        var_name="time_col",
        value_name="raw_metrics"
    )

    melted["timestamp"] = melted["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))

    parsed = melted["raw_metrics"].apply(parse_metrics_c_style)
    met = pd.DataFrame(parsed.tolist(), columns=["sales", "orders", "ads", "ads_ro"])

    out = pd.concat([melted[["timestamp"] + id_cols], met], axis=1)
    out.rename(columns={"name": "channel"}, inplace=True)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    for c in ["sales", "orders", "ads", "ads_ro"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# -------------------- Data source & cache --------------------
@st.cache_data(ttl=600, show_spinner=False)  # 10 นาที
def fetch_csv() -> pd.DataFrame:
    url = os.environ.get("ROAS_CSV_URL", "").strip()
    if not url:
        raise RuntimeError("ROAS_CSV_URL is not set in Secrets.")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

@st.cache_data(ttl=600, show_spinner=False)
def build_long() -> pd.DataFrame:
    df_wide = fetch_csv()
    return wide_to_long(df_wide)

def clear_all_cache():
    fetch_csv.clear()
    build_long.clear()

# -------------------- Common utils --------------------
def hourly_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d["timestamp"].dt.floor("H")
    idx = d.sort_values("timestamp").groupby(["channel", "hour"]).tail(1).index
    return d.loc[idx].reset_index(drop=True)

def totals_from_df(df: pd.DataFrame):
    t = {}
    t["Sales"]  = df["sales"].sum()
    t["Orders"] = df["orders"].sum()
    t["Ads"]    = df["ads"].sum()

    # ads_ro = mean เฉพาะค่าที่ > 0
    pos = df["ads_ro"].dropna()
    pos = pos[pos > 0]
    t["ads_ro"] = pos.mean() if len(pos) else np.nan

    # sale_ro = sum(sales)/sum(ads)
    t["sale_ro"] = (t["Sales"] / t["Ads"]) if t["Ads"] else np.nan
    return t

def pct_delta(curr, prev):
    if prev is None or np.isnan(prev) or prev == 0:
        return None
    if curr is None or np.isnan(curr):
        return None
    return (curr - prev) * 100.0 / prev

def metric_card(col, label, value, ref_value):
    delta = pct_delta(value, ref_value)
    if value is None or np.isnan(value):
        val_txt = "-"
    else:
        # ขึ้นรูปแบบสวยๆ ถ้าเป็นอัตราเป็นทศนิยม 2 ตำแหน่ง
        if label in ["ads_ro", "sale_ro"]:
            val_txt = f"{value:,.2f}"
        else:
            val_txt = f"{value:,.0f}"
    if delta is None:
        col.metric(label, val_txt)
    else:
        col.metric(label, val_txt, f"{delta:+.1f}%")

def default_date_range(df_hourly):
    # ค่า default = 3 วันล่าสุด
    if df_hourly.empty:
        today = pd.Timestamp.now(tz="Asia/Bangkok").date()
        return today - timedelta(days=2), today
    max_day = df_hourly["hour"].max().date()
    start = max_day - timedelta(days=2)
    return start, max_day

def overlay_dataset(df_hourly, start_date, end_date):
    """
    คืน dataframe สำหรับทำกราฟ overlay:
    one row = (day, HH:MM, metric aggregates by day-hour)
    """
    d = df_hourly[(df_hourly["hour"].dt.date >= start_date) & (df_hourly["hour"].dt.date <= end_date)].copy()
    if d.empty:
        return pd.DataFrame()

    d["day"] = d["hour"].dt.date
    d["hm"] = d["hour"].dt.strftime("%H:%M")

    # aggregate per day/hour across channels:
    g = d.groupby(["day", "hm"]).agg(
        sales=("sales", "sum"),
        orders=("orders", "sum"),
        ads=("ads", "sum"),
        ads_ro=("ads_ro", lambda s: s[s > 0].mean() if len(s.dropna()) else np.nan)
    ).reset_index()

    # คำนวณ sale_ro ต่อ hour: sum(sales)/sum(ads)
    sale_sum = g.groupby(["day", "hm"])["sales"].transform("sum")
    ads_sum = g.groupby(["day", "hm"])["ads"].transform("sum")
    g["sale_ro"] = sale_sum / ads_sum.replace(0, np.nan)

    return g.sort_values(["day", "hm"])

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Filters")
    reload_btn = st.button("Reload", use_container_width=True)
    if reload_btn:
        clear_all_cache()
        st.rerun()

# -------------------- Load data --------------------
try:
    df_long = build_long()
except Exception as e:
    st.error(f"Load/parse error: {e}")
    st.stop()

# Hourly snapshot (latest record in each hour per channel)
df_hourly = hourly_latest_snapshot(df_long)

# Default date range = 3 วันล่าสุด
with st.sidebar:
    s_default, e_default = default_date_range(df_hourly)
    dr = st.date_input("Date range (default 3 days)", value=(s_default, e_default))
    if isinstance(dr, (list, tuple)):
        date_start, date_end = dr
    else:
        # กรณีเลือกวันเดียว
        date_start, date_end = dr, dr

# Channel filter (เลือก "All" ไว้ให้): ใช้ใน Overview/Compare
with st.sidebar:
    channels_all = sorted(df_hourly["channel"].dropna().unique().tolist())
    sel = st.multiselect("Channels (เลือก All ได้)", options=["[All]"] + channels_all, default=["[All]"])
    if "[All]" in sel:
        selected_channels = channels_all
    else:
        selected_channels = sel

# -------------------- Page router --------------------
with st.sidebar:
    page = st.radio("Page", ["Overview", "Channel", "Compare"])

st.title("Shopee ROAS Dashboard")
st.caption(f"Last refresh: {pd.Timestamp.now(tz='Asia/Bangkok').strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------- KPI header (common function) --------------------
def render_kpi_header(df_hourly_scope):
    # scope = ช่วงวัน + ช่องที่เลือก + hourly latest snapshot
    today_mask = (df_hourly_scope["hour"].dt.date >= date_start) & (df_hourly_scope["hour"].dt.date <= date_end)
    curr = df_hourly_scope.loc[today_mask].copy()

    # baseline = เมื่อวานช่วงเดียวกัน
    y_start = (pd.Timestamp(date_start) - timedelta(days=1)).date()
    y_end   = (pd.Timestamp(date_end)   - timedelta(days=1)).date()
    base_mask = (df_hourly_scope["hour"].dt.date >= y_start) & (df_hourly_scope["hour"].dt.date <= y_end)
    base = df_hourly_scope.loc[base_mask].copy()

    curr_tot = totals_from_df(curr) if not curr.empty else {}
    base_tot = totals_from_df(base) if not base.empty else {}

    cols = st.columns(5)
    metric_card(cols[0], "Sales",  curr_tot.get("Sales", np.nan),  base_tot.get("Sales", np.nan))
    metric_card(cols[1], "Orders", curr_tot.get("Orders", np.nan), base_tot.get("Orders", np.nan))
    metric_card(cols[2], "Ads",    curr_tot.get("Ads", np.nan),    base_tot.get("Ads", np.nan))
    metric_card(cols[3], "ads_ro", curr_tot.get("ads_ro", np.nan), base_tot.get("ads_ro", np.nan))
    metric_card(cols[4], "sale_ro",curr_tot.get("sale_ro", np.nan),base_tot.get("sale_ro", np.nan))

# -------------------- OVERVIEW --------------------
if page == "Overview":
    scope = df_hourly[df_hourly["channel"].isin(selected_channels)].copy()

    st.subheader("Overview (All selected channels)")
    render_kpi_header(scope)

    # ---------- Overlay chart ----------
    st.markdown("### Trend overlay by day")
    overlay = overlay_dataset(scope, date_start, date_end)
    metric_label = st.selectbox(
        "Metric to plot (เลือกได้ 1 ค่า)",
        options=["Sales", "Orders", "Ads", "ads_ro", "sale_ro"],
        index=0
    )
    key_map = {"Sales":"sales","Orders":"orders","Ads":"ads","ads_ro":"ads_ro","sale_ro":"sale_ro"}

    if overlay.empty:
        st.info("No data for the selected filters/dates.")
    else:
        fig = go.Figure()
        for day, sub in overlay.groupby("day"):
            fig.add_trace(go.Scatter(
                x=sub["hm"], y=sub[key_map[metric_label]],
                mode="lines+markers", name=str(day)
            ))
        fig.update_layout(
            xaxis_title="Time (HH:MM)",
            yaxis_title=metric_label,
            legend=dict(orientation="h", y=-0.25),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Table (same data as chart) ----------
        st.markdown("### Data (hourly latest snapshot aggregated)")
        show_cols = ["day","hm","sales","orders","ads","ads_ro","sale_ro"]
        t = overlay[show_cols].copy()
        try:
            import matplotlib  # need for background_gradient
            sty = t.style.background_gradient(subset=["sales","orders","ads","ads_ro","sale_ro"])
            st.dataframe(sty, use_container_width=True, height=380)
        except Exception:
            st.dataframe(t, use_container_width=True, height=380)

# -------------------- CHANNEL --------------------
elif page == "Channel":
    st.subheader("Channel")
    # ให้เลือกได้จาก "ทุกช่องในชีต" ต่อให้ sidebar ไม่ได้เลือก
    chan = st.selectbox("Pick one channel", options=sorted(df_hourly["channel"].unique()))
    scope = df_hourly[df_hourly["channel"] == chan].copy()

    render_kpi_header(scope)

    st.markdown("### Trend overlay by day (channel)")
    overlay = overlay_dataset(scope, date_start, date_end)
    metric_label = st.selectbox(
        "Metric to plot (เลือกได้ 1 ค่า)", options=["Sales","Orders","Ads","ads_ro","sale_ro"], index=0, key="ch_metric"
    )
    key_map = {"Sales":"sales","Orders":"orders","Ads":"ads","ads_ro":"ads_ro","sale_ro":"sale_ro"}

    if overlay.empty:
        st.info("No data for the selected range.")
    else:
        fig = go.Figure()
        for day, sub in overlay.groupby("day"):
            fig.add_trace(go.Scatter(
                x=sub["hm"], y=sub[key_map[metric_label]],
                mode="lines+markers", name=str(day)
            ))
        fig.update_layout(
            xaxis_title="Time (HH:MM)",
            yaxis_title=metric_label,
            legend=dict(orientation="h", y=-0.25),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Data (hourly latest snapshot aggregated)")
        show_cols = ["day","hm","sales","orders","ads","ads_ro","sale_ro"]
        t = overlay[show_cols].copy()
        try:
            import matplotlib
            sty = t.style.background_gradient(subset=["sales","orders","ads","ads_ro","sale_ro"])
            st.dataframe(sty, use_container_width=True, height=380)
        except Exception:
            st.dataframe(t, use_container_width=True, height=380)

# -------------------- COMPARE --------------------
else:
    st.subheader("Compare channels")
    pick = st.multiselect("Pick 2–4 channels", options=sorted(df_hourly["channel"].unique()), default=sorted(df_hourly["channel"].unique())[:2])
    if len(pick) < 2:
        st.info("Please pick at least 2 channels.")
        st.stop()

    sub = df_hourly[df_hourly["channel"].isin(pick)].copy()
    m = sub[(sub["hour"].dt.date >= date_start) & (sub["hour"].dt.date <= date_end)]

    # KPI table by channel (mean/ratio)
    def _kpi_df(x):
        t = totals_from_df(x)
        return pd.Series({
            "Sales": t["Sales"],
            "Orders": t["Orders"],
            "Ads": t["Ads"],
            "ads_ro": t["ads_ro"],
            "sale_ro": t["sale_ro"],
        })

    tbl = m.groupby("channel").apply(_kpi_df).reset_index()
    st.markdown("### KPI comparison table")
    try:
        import matplotlib
        st.dataframe(tbl.style.background_gradient(subset=["Sales","Orders","Ads","ads_ro","sale_ro"]),
                     use_container_width=True)
    except Exception:
        st.dataframe(tbl, use_container_width=True)

    # Overlay chart (เลือก metric 1 ค่า)
    st.markdown("### Trend overlay by day")
    metric_label = st.selectbox("Metric", ["Sales","Orders","Ads","ads_ro","sale_ro"], index=0, key="cmp_metric")
    key_map = {"Sales":"sales","Orders":"orders","Ads":"ads","ads_ro":"ads_ro","sale_ro":"sale_ro"}

    fig = go.Figure()
    for ch in pick:
        ch_scope = m[m["channel"] == ch]
        ov = overlay_dataset(ch_scope, date_start, date_end)
        for day, subd in ov.groupby("day"):
            fig.add_trace(go.Scatter(
                x=subd["hm"], y=subd[key_map[metric_label]],
                mode="lines+markers", name=f"{ch} • {day}"
            ))

    fig.update_layout(
        xaxis_title="Time (HH:MM)",
        yaxis_title=metric_label,
        height=440,
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True)
