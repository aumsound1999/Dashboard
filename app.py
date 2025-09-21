# app.py
import os, io, re, unicodedata
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(page_title="Shopee ROAS", layout="wide")

# =========================
# Header cleaning (key patch)
# =========================
def clean_header(s: str) -> str:
    """
    ทำความสะอาดชื่อคอลัมน์ให้เหลือ pattern ที่อ่านได้แน่นอน
    - normalize unicode
    - ลบ zero-width, BOM, NBSP ฯลฯ
    - บีบ whitespace ให้เหลือ space ปกติ
    - เก็บเฉพาะ [a-zA-Z0-9] ช่องว่าง และ ':'
    """
    s = unicodedata.normalize("NFKC", str(s))
    for bad in ["\xa0", "\u00A0", "\u202F", "\u2007", "\u2009",
                "\ufeff", "\u200b", "\u200c", "\u200d"]:
        s = s.replace(bad, " ")
    s = "".join(ch for ch in s if ch.isalnum() or ch in " :")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# Flexible time-column detector (key patch)
# =========================
# e.g. "D21 15:55" / "D 21 15:55" / "d21 09:07"
TIME_COL_PATTERN = re.compile(r"^[A-Za-z]\s*\d{1,2}\s+\d{1,2}:\d{1,2}$")

def is_time_col(col: str) -> bool:
    c = clean_header(col)
    return TIME_COL_PATTERN.match(c) is not None

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

# =========================
# Metric parser
# =========================
def str_to_metrics_tuple(s: str):
    """
    cell รูปแบบ "budget,user,order,view,sale,ro,..." -> เอาเฉพาะ 6 ตัวแรก
    ถ้าไม่ใช่ตัวเลข ให้ NaN
    """
    if not isinstance(s, str):
        return (np.nan,)*6
    if not re.search(r"\d", s):
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

# =========================
# Wide -> Long
# =========================
def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # ทำความสะอาดชื่อคอลัมน์ทั้งหมดก่อน
    df_wide = df_wide.rename(columns=lambda c: clean_header(c))

    cols = list(df_wide.columns)
    id_cols, time_cols = [], []
    for c in cols:
        if clean_header(c).lower() == "name":
            id_cols.append(c)
        elif is_time_col(c):
            time_cols.append(c)
        else:
            pass

    if not time_cols:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:4'.")

    df_melt = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                           var_name="time_col", value_name="metrics")

    # timestamp
    ts = df_melt["time_col"].apply(lambda x: parse_timestamp_from_header(x, tz=tz))
    df_melt["timestamp"] = ts

    # metrics
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

# =========================
# Data loader (from Secret URL)
# =========================
@st.cache_data(ttl=60*10, show_spinner=False)  # cache 10 นาที
def fetch_sheet():
    url = os.environ.get("ROAS_CSV_URL", "").strip()
    if not url:
        raise RuntimeError("Missing secret ROAS_CSV_URL")

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # ทำความสะอาด header ทันที (กันตาย)
    df.rename(columns=lambda c: clean_header(c), inplace=True)
    return df

@st.cache_data(ttl=60*10, show_spinner=False)
def build_long(df):
    return long_from_wide(df)

# =========================
# UI
# =========================
st.title("Shopee ROAS Dashboard")

left, right = st.columns([1, 3])
with left:
    if st.button("Reload", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Last refresh: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# โหลดข้อมูล
try:
    df_wide = fetch_sheet()
except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# แปลงเป็น long
try:
    df_long = build_long(df_wide)
except Exception as e:
    st.error(f"Parse failed: {e}")
    st.dataframe(df_wide.head(), use_container_width=True)
    st.stop()

# ============ Filters ============
st.sidebar.header("Filters")
channels_all = sorted(df_long["channel"].dropna().unique().tolist())
default_days = 3
tz = "Asia/Bangkok"
max_ts = df_long["timestamp"].max()
min_ts = df_long["timestamp"].min()
if pd.isna(max_ts) or pd.isna(min_ts):
    st.warning("No valid timestamps found.")
    st.stop()

start_def = (max_ts - pd.Timedelta(days=default_days-1)).date()
end_def   = max_ts.date()
d1, d2 = st.sidebar.date_input(
    "Date range (default 3 days)", value=(start_def, end_def),
    min_value=min_ts.date(), max_value=max_ts.date()
)
if isinstance(d1, (list, tuple)):  # streamlit edge-case
    d1, d2 = d1

channel_pick = st.sidebar.multiselect(
    "Channels (เลือก All ได้)",
    options=["[All]"] + channels_all,
    default=["[All]"]
)

# สร้าง mask
start_ts = pd.Timestamp.combine(d1, pd.Timestamp("00:00").time()).tz_localize(tz)
end_ts   = pd.Timestamp.combine(d2, pd.Timestamp("23:59").time()).tz_localize(tz)

if "[All]" in channel_pick or len(channel_pick) == 0:
    mask_ch = df_long["channel"].isin(channels_all)
else:
    mask_ch = df_long["channel"].isin(channel_pick)

mask = mask_ch & (df_long["timestamp"].between(start_ts, end_ts))

d = df_long.loc[mask].copy()
if d.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# baseline = yesterday same clock range
y_start = start_ts - pd.Timedelta(days=1)
y_end   = end_ts   - pd.Timedelta(days=1)
base_mask = mask_ch & (df_long["timestamp"].between(y_start, y_end))
baseline = df_long.loc[base_mask].copy()

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

# ------- Overview KPIs (ดูจากคอลัมน์ C ตามตรรกะในตาราง) -------
# จากไฟล์: C คือ snapshot ล่าสุดของแต่ละแถว (ชั่วโมงปัจจุบัน)
# เรา aggregate ด้วยการเลือก snapshot ล่าสุดต่อช่อง ต่อชั่วโมง => แล้ว sum

def latest_per_hour(df):
    x = df.copy()
    x["hour"] = x["timestamp"].dt.floor("H")
    idx = x.sort_values("timestamp").groupby(["channel","hour"]).tail(1).index
    return x.loc[idx]

hourly = latest_per_hour(d)
baseline_hourly = latest_per_hour(baseline) if not baseline.empty else None

tot = compute_totals(hourly)
base_tot = compute_totals(baseline_hourly) if baseline_hourly is not None else {}

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

st.markdown("### Trend by hour (Sales/Orders/ROAS)")
trend = hourly.groupby("hour").agg({"sale":"sum","order":"sum","budget":"sum"}).reset_index()
trend["ROAS"] = trend["sale"] / trend["budget"].replace(0, np.nan)
fig = go.Figure()
fig.add_trace(go.Scatter(x=trend["hour"], y=trend["sale"],   name="Sales",  mode="lines+markers"))
fig.add_trace(go.Scatter(x=trend["hour"], y=trend["order"],  name="Orders", mode="lines+markers", yaxis="y2"))
fig.add_trace(go.Scatter(x=trend["hour"], y=trend["ROAS"],   name="ROAS",  mode="lines+markers", yaxis="y3"))
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
