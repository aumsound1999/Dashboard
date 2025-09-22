# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Shopee ROAS Dashboard",
    layout="wide",
    page_icon="📈"
)

# =========================================================
# Utilities
# =========================================================
def hhmm_str(h):
    """int hour -> 'HH:MM'"""
    try:
        h = int(h)
    except Exception:
        return str(h)
    return f"{h:02d}:00"


def normalize_hour_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df['hour_str'] exists (HH:MM)."""
    out = df.copy()
    if "hour_str" in out.columns:
        return out

    if "hour" not in out.columns:
        raise ValueError("DataFrame must have 'hour' column")

    h = out["hour"]
    if np.issubdtype(h.dtype, np.number):
        out["hour_str"] = h.astype(int).clip(0, 23).map(hhmm_str)
    elif np.issubdtype(h.dtype, np.datetime64):
        out["hour_str"] = pd.to_datetime(h).dt.strftime("%H:%M")
    else:
        out["hour_str"] = h.astype(str)
    return out


def compute_ro_series(df: pd.DataFrame, top_col: str, ads_col: str, cap=50.0) -> pd.Series:
    """
    RO = Δtop / Δads (คำนวณเป็นรายชั่วโมง "ภายในวันเดียวกัน")
    - ถ้า Δads<=0 -> NaN
    - cap ค่า MAX ที่ 50
    """
    ds = df.groupby("day")[top_col].diff()
    da = df.groupby("day")[ads_col].diff()
    ro = np.where(da > 0, ds / da, np.nan)
    return pd.Series(np.clip(ro, None, cap))


def hour_diff(df: pd.DataFrame, col: str) -> pd.Series:
    """Δรายชั่วโมงภายในวันเดียวกัน (ไม่ข้ามวัน)"""
    return df.groupby("day")[col].diff()


# =========================================================
# (Demo) Data generator — ใช้งานได้ทันที
# *** เปลี่ยนเป็น loader ของข้อมูลจริงของคุณได้ตามสะดวก ***
# =========================================================
@st.cache_data(show_spinner=False)
def get_demo_data(days_back=3, n_channels=3, seed=3):
    """
    สร้างข้อมูลเดโม่ให้ใช้งานได้ทันที:
      - hourly_overlay: ['day','hour','channel','sales','orders','ads','ads_sales']
      - credits: ตารางยอดเครดิตคงเหลือ & จำนวนแคมเปญที่เปิด
    """
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.now().floor("H")
    days = [ (today - timedelta(days=i)).date().isoformat() for i in range(days_back, -1, -1) ]
    hours = list(range(0, 24))

    channels = [f"ch_{i+1:02d}" for i in range(n_channels)]
    records = []
    for d in days:
        for h in hours:
            for ch in channels:
                # simulate monotonic-ish cumulative metrics
                base = (int(d[-2:]) * 17 + h * 13 + (hash(ch) % 29)) % 100
                sales = max(0, base * 120 + rng.normal(0, 200))
                orders = max(0, base * 0.6 + rng.normal(0, 3))
                ads = max(0, base * 1.8 + rng.normal(0, 8))
                # สมมุติยอดขายจากแอด (ถ้าไม่มีคอลัมน์นี้ในข้อมูลจริง ให้ลบทิ้งและจะ fallback เป็น sales)
                ads_sales = max(0, base * 90 + rng.normal(0, 120))
                records.append([d, h, ch, sales, orders, ads, ads_sales])

    hourly_overlay = pd.DataFrame(
        records,
        columns=["day", "hour", "channel", "sales", "orders", "ads", "ads_sales"]
    )

    # เดโม่เครดิต
    cred = pd.DataFrame({
        "channel": channels,
        "credits_left": rng.integers(100, 1200, size=len(channels)),
        "active_campaigns": rng.integers(0, 3, size=len(channels)),
    })

    return hourly_overlay, cred


# =========================================================
# TODO: เปลี่ยนจาก get_demo_data() เป็น Loader ของจริง
# =========================================================
hourly_overlay, credits_df = get_demo_data(days_back=3, n_channels=5, seed=42)

# =========================================================
# Sidebar filters
# =========================================================
st.sidebar.markdown("### Filters")

# Date range (demo ไม่ได้ตัดข้อมูลตามวันจริง ถ้าข้อมูลจริงมี timestamp ให้ filter ที่นี่)
default_start = (datetime.now() - timedelta(days=2)).date()
default_end = datetime.now().date()
date_range = st.sidebar.date_input(
    "Date range (default 3 days)",
    value=(default_start, default_end)
)

channels = sorted(hourly_overlay["channel"].unique())
selected_channels = st.sidebar.multiselect(
    "Channels (เลือก All ได้)",
    options=channels,
    default=channels
)

# Filter data ตาม channel
df = hourly_overlay[hourly_overlay["channel"].isin(selected_channels)].copy()
df = normalize_hour_col(df)

# =========================================================
# KPI (สรุป)
# =========================================================
def kpi_block(df: pd.DataFrame):
    # รวมยอดล่าสุดต่อวัน (เอาชั่วโมงสูงสุดของแต่ละวัน)
    df_last = df.sort_values(["day", "hour"]).groupby(["day", "channel"], as_index=False).last()
    # รวมทุกช่อง = KPI รวม
    grp = df_last.groupby("day", as_index=False)[["sales", "orders", "ads"]].sum()
    grp = grp.sort_values("day")

    # เปรียบเทียบกับวันก่อนหน้า
    def pct_delta(s):
        if len(s) < 2:
            return 0.0
        prev, cur = s.iloc[-2], s.iloc[-1]
        if prev == 0:
            return 0.0
        return (cur - prev) / prev * 100.0

    total_sales = grp["sales"].iloc[-1] if len(grp) else 0.0
    total_orders = grp["orders"].iloc[-1] if len(grp) else 0.0
    total_ads = grp["ads"].iloc[-1] if len(grp) else 0.0

    # sale_ro (รวมวันล่าสุดด้วยสูตร per-hour diff รวม)
    tmp = df.sort_values(["day","hour"])
    sale_ro_series = compute_ro_series(tmp, top_col="sales", ads_col="ads")
    tmp2 = tmp.assign(sale_ro=sale_ro_series)
    sale_ro_avg = np.nanmean(tmp2[tmp2["day"]==tmp2["day"].max()]["sale_ro"].to_numpy())

    # ads_ro (ถ้ามี ads_sales จะใช้ ads_sales; ถ้าไม่มี fallback sales)
    ads_top = "ads_sales" if "ads_sales" in df.columns else "sales"
    ads_ro_series = compute_ro_series(tmp, top_col=ads_top, ads_col="ads")
    tmp3 = tmp.assign(ads_ro=ads_ro_series)
    ads_ro_avg = np.nanmean(tmp3[tmp3["day"]==tmp3["day"].max()]["ads_ro"].to_numpy())

    sales_delta = pct_delta(grp["sales"])
    orders_delta = pct_delta(grp["orders"])
    ads_delta = pct_delta(grp["ads"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Sales", f"{total_sales:,.0f}", f"{sales_delta:+.1f}%")
    with c2:
        st.metric("Orders", f"{total_orders:,.0f}", f"{orders_delta:+.1f}%")
    with c3:
        st.metric("Budget (Ads)", f"{total_ads:,.0f}", f"{ads_delta:+.1f}%")
    with c4:
        st.metric("sale_ro (Sales/Ads)", f"{sale_ro_avg:,.3f}")
    with c5:
        st.metric("ads_ro (avg>0)", f"{ads_ro_avg:,.2f}")

# =========================================================
# Trend by hour (เลือก metric ได้)
# =========================================================
def trend_by_hour(df: pd.DataFrame):
    st.markdown("### Trend by hour")

    METRIC_OPTIONS = {
        "sales": "Sales",
        "orders": "Orders",
        "ads": "Budget (Ads)",
        "sale_ro": "sale_ro (Sales/Ads — per-hour diff, capped 50)",
        "ads_ro": "ads_ro (AdsSales/Ads — per-hour diff, capped 50)",
    }

    if "metric_hour" not in st.session_state:
        st.session_state.metric_hour = "sales"

    metric_key = st.selectbox(
        "Metric to plot (เลือกได้ 1 ค่า)",
        list(METRIC_OPTIONS.keys()),
        format_func=lambda k: METRIC_OPTIONS[k],
        index=list(METRIC_OPTIONS.keys()).index(st.session_state.metric_hour)
    )
    st.session_state.metric_hour = metric_key

    df_plot = df.copy()
    df_plot = normalize_hour_col(df_plot)
    df_plot = df_plot.sort_values(["day", "hour_str"])

    if metric_key in ("sales", "orders", "ads"):
        df_plot["y"] = df_plot[metric_key].astype(float)
    elif metric_key == "sale_ro":
        df_plot["y"] = compute_ro_series(df_plot, top_col="sales", ads_col="ads")
    else:  # ads_ro
        ads_top = "ads_sales" if "ads_sales" in df_plot.columns else "sales"
        df_plot["y"] = compute_ro_series(df_plot, top_col=ads_top, ads_col="ads")

    fig = px.line(
        df_plot,
        x="hour_str",
        y="y",
        color="day",
        markers=True,
        line_shape="linear",
        labels={"hour_str": "Hour (HH:MM)", "y": METRIC_OPTIONS[st.session_state.metric_hour], "day": "Date"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Day")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Prime hours heatmap (เลือก metric เดียวกัน โดยใช้ per-hour diff สำหรับ sales/orders/ads)
# =========================================================
def prime_hours_heatmap(df: pd.DataFrame, metric_key: str):
    st.markdown("### Prime hours heatmap")

    dfh = df.copy()
    dfh = normalize_hour_col(dfh)
    dfh = dfh.sort_values(["day", "hour_str"])

    if metric_key in ("sales", "orders", "ads"):
        val = hour_diff(dfh, metric_key)  # ใช้ Δ ต่อชั่วโมงในวันเดียวกันเพื่อหา prime hour
        title = f"{metric_key} (Δhourly)"
    elif metric_key == "sale_ro":
        val = compute_ro_series(dfh, top_col="sales", ads_col="ads")
        title = "sale_ro (Δsales/Δads, capped 50)"
    else:
        ads_top = "ads_sales" if "ads_sales" in dfh.columns else "sales"
        val = compute_ro_series(dfh, top_col=ads_top, ads_col="ads")
        title = "ads_ro (Δads_sales/Δads, capped 50)"

    dfh = dfh.assign(val=val)
    # pivot day x hour
    pivot = dfh.pivot_table(index="day", columns="hour_str", values="val", aggfunc="mean")
    pivot = pivot.sort_index(axis=1)  # sort columns HH:MM

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Day", color="value"),
        aspect="auto",
        color_continuous_scale="Blues",
        origin="lower"
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Advertising credits low
# =========================================================
def low_credits_table(credits: pd.DataFrame):
    st.markdown("### Advertising credits are low")

    # กติกา: เครดิต < 500 และมีแคมเปญเปิดอย่างน้อย 1
    cond = (credits["credits_left"] < 500) & (credits["active_campaigns"] > 0)
    low = credits.loc[cond, ["channel", "credits_left", "active_campaigns"]].copy()
    low = low.sort_values(["credits_left", "active_campaigns"])

    if low.empty:
        st.info("ไม่มีช่องที่เครดิตต่ำกว่า 500 และมีแคมเปญที่เปิดใช้งาน")
        return

    low.columns = ["ช่อง", "เครดิตเหลือ", "แคมที่เปิด"]
    st.dataframe(
        low,
        use_container_width=True,
        hide_index=True
    )
    st.caption("เงื่อนไข: เครดิต < 500 และมีแคมเปญเปิดอย่างน้อย 1 แคม")


# =========================================================
# Layout
# =========================================================
st.title("Shopee ROAS Dashboard")

st.caption(f"Last refresh: {datetime.now():%Y-%m-%d %H:%M:%S}")

st.markdown("#### Overview (All selected channels)")
kpi_block(df)

# Trend + Heatmap
trend_by_hour(df)
prime_hours_heatmap(df, st.session_state.metric_hour)

# Low credits
low_credits_table(credits_df)
