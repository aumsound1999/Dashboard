# app.py
# Shopee ROAS Dashboard — Overview • Channel • Compare
# อ่านข้อมูลจาก Google Sheet (CSV export) ผ่าน Secrets:
#    ROAS_CSV_URL="https://docs.google.com/spreadsheets/d/<ID>/gviz/tqx=out=csv&sheet=<SHEET>"
#
# pip: streamlit pandas numpy plotly requests

import os
import io
import re
import ast
from datetime import timedelta, date as date_type
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Shopee ROAS", layout="wide")

# IMPROVEMENT: ใช้ Constants เพื่อให้อ่านง่ายและแก้ไขสะดวก
METRIC_COLUMNS = ["sales", "orders", "ads", "view", "ads_ro_raw", "misc"]
V_COLUMNS = [f"v{i}" for i in range(len(METRIC_COLUMNS))]

# --- NEW: Define staff and their channels ---
STAFF_CHANNELS = {
    "jen": [
        "luckyrich_mart", "prosperway_store", "richvibe_market", "starshop_789",
        "lovely_cattt", "patchalady5555", "monostore_99", "helloo1999", "shopletter"
    ],
    "fon": [
        "siri_lovepink23", "goldenflow_shop", "cherry.good", "infinity_jj",
        "mocha_winkkk", "shop.sabai_", "sale_mak_mak", "pop_py_mart",
        "miniyou_01", "bubbu_town777"
    ]
}

# -----------------------------------------------------------------------------
# Helpers: detect & parse
# -----------------------------------------------------------------------------

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$") # e.g., D21 12:45

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamps_from_headers(headers: list[str], tz: str = "Asia/Bangkok") -> dict:
    """
    CRITICAL FIX: แก้ไข Logic การอ่านเวลาให้ถูกต้องตามโครงสร้างไฟล์
    - ยึดคอลัมน์เวลา 'แรกสุด' (ซ้ายสุด) เป็นข้อมูลล่าสุด
    - คำนวณเวลาย้อนหลังไปยังคอลัมน์ทางขวา
    """
    timestamps = {}
    now = pd.Timestamp.now(tz=tz)
    
    # Get only the columns that match the time format, in their original order
    time_cols_only = [h for h in headers if is_time_col(h)]
    if not time_cols_only:
        return {h: pd.NaT for h in headers}

    temp_timestamps = {}
    previous_ts = pd.NaT

    # Iterate FORWARDS through the time columns, as the first one is the latest
    for hdr in time_cols_only:
        hdr_strip = hdr.strip()
        m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr_strip)
        if not m:
            continue
        
        d, hh, mm = map(int, m.groups())
        
        if pd.isna(previous_ts):
            # ANCHORING LOGIC: This runs only for the very first time column found
            # Assume its day number 'd' refers to a date at or before 'now'.
            # Start with today and go backwards until we find a matching day number.
            anchor_date_candidate = now
            for _ in range(45): # Safety break after 45 days
                if anchor_date_candidate.day == d:
                    break
                anchor_date_candidate -= pd.Timedelta(days=1)
            else:
                temp_timestamps[hdr] = pd.NaT
                continue
            
            try:
                ts = pd.Timestamp(year=anchor_date_candidate.year, month=anchor_date_candidate.month, day=d, hour=hh, minute=mm, tz=tz)
                temp_timestamps[hdr] = ts
                previous_ts = ts
            except ValueError:
                temp_timestamps[hdr] = pd.NaT

        else:
            # ITERATION LOGIC: For all subsequent (older) time columns
            try:
                # Tentatively create a timestamp using the year/month of the previous (newer) data point
                ts = pd.Timestamp(year=previous_ts.year, month=previous_ts.month, day=d, hour=hh, minute=mm, tz=tz)
                
                # The current timestamp must be EARLIER than the previous one.
                # If it's not, it means we've crossed a month boundary going backwards in time.
                if ts >= previous_ts:
                    ts -= pd.DateOffset(months=1)
                
                temp_timestamps[hdr] = ts
                previous_ts = ts
            except ValueError:
                # This can happen if a day 'd' doesn't exist in the guessed month (e.g. 31 in Feb)
                try:
                    prev_month_date = previous_ts - pd.DateOffset(months=1)
                    ts = pd.Timestamp(year=prev_month_date.year, month=prev_month_date.month, day=d, hour=hh, minute=mm, tz=tz)
                    temp_timestamps[hdr] = ts
                    previous_ts = ts
                except ValueError:
                    temp_timestamps[hdr] = pd.NaT

    # Map the parsed timestamps back to the full list of original headers
    for hdr in headers:
        timestamps[hdr] = temp_timestamps.get(hdr, pd.NaT)
        
    return timestamps


def parse_metrics_cell(s: str):
    """
    FIX: ทำให้การ parse ข้อมูลมีความแม่นยำมากขึ้น โดยจัดการกับค่าที่หายไป (,,) ได้ถูกต้อง
    """
    if not isinstance(s, str):
        return [np.nan] * 6
    
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    # ไม่ลบ empty string ออก เพื่อรักษาตำแหน่งของข้อมูล
    parts = s_clean.split(",")
    nums = []
    
    # วนลูปตามจำนวนค่าที่คาดหวัง (6 ค่า)
    for p in parts[:6]:
        # แปลงค่าว่างให้เป็น NaN
        if p.strip() == "":
            nums.append(np.nan)
            continue
        try:
            nums.append(float(p))
        except (ValueError, TypeError):
            nums.append(np.nan)
    
    # เติม NaN หากข้อมูลที่มาสั้นกว่า 6 ค่า
    while len(nums) < 6:
        nums.append(np.nan)
    return nums

def parse_campaign_details(campaign_string: str):
    """
    [CORRECTED] อัปเกรดฟังก์ชันให้สามารถอ่านข้อมูลแคมเปญในรูปแบบ Dictionary (JSON) ใหม่ได้
    """
    if not isinstance(campaign_string, str):
        return []

    try:
        # ast.literal_eval สามารถแปลง String ที่หน้าตาเหมือน Dictionary ของ Python ได้
        data_dict = ast.literal_eval(campaign_string)
        
        # ตรวจสอบว่าเป็น Dictionary และมี key 'campaigns' อยู่ข้างในหรือไม่
        if isinstance(data_dict, dict) and 'campaigns' in data_dict:
            # ดึง list ของ campaign ออกมา ซึ่งอาจจะเป็น list ว่างก็ได้
            return data_dict.get('campaigns', [])
        else:
            # ถ้าโครงสร้างไม่ถูกต้อง ให้คืนค่าเป็น list ว่าง
            return []
            
    except (ValueError, SyntaxError):
        # จัดการกรณีที่ String ไม่ใช่รูปแบบที่ถูกต้อง
        return []

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Fetching latest data...")
def fetch_csv_text():
    # --- CORRECTED: Read from os.environ for Hugging Face ---
    url = os.environ.get("ROAS_CSV_URL", "") # Read from environment variable
    if not url:
        raise RuntimeError("Missing Secrets: ROAS_CSV_URL is not set in Hugging Face secrets.")
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=600, show_spinner=False)
def load_wide_df():
    csv_text = fetch_csv_text()
    return pd.read_csv(io.StringIO(csv_text))

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    # IMPROVEMENT: Automatically detect all ID columns before the first time column
    first_time_col_index = -1
    for i, col in enumerate(df_wide.columns):
        if is_time_col(str(col)):
            first_time_col_index = i
            break
            
    if first_time_col_index == -1:
        raise ValueError("No time columns detected. Expect headers like 'D21 12:45'.")
        
    id_cols = df_wide.columns[:first_time_col_index].tolist()
    time_cols = [col for col in df_wide.columns[first_time_col_index:] if is_time_col(str(col))]

    if not id_cols:
         raise ValueError("No identifier columns (like 'name') found before time columns.")
         
    ts_map = parse_timestamps_from_headers(df_wide.columns, tz=tz)

    m = df_wide.melt(id_vars=id_cols, value_vars=time_cols,
                       var_name="time_col", value_name="raw")
    m["timestamp"] = m["time_col"].map(ts_map)

    V = pd.DataFrame(m["raw"].apply(parse_metrics_cell).tolist(), columns=V_COLUMNS)
    
    # Build rename dictionary for known and potential ID columns
    rename_dict = {}
    if len(id_cols) > 0:
        rename_dict[id_cols[0]] = 'channel' # Assume first is always channel
    if len(id_cols) > 1:
        rename_dict[id_cols[1]] = 'campaign' # Assume second is campaign

    out = pd.concat([m[["timestamp"] + id_cols], V], axis=1).rename(columns=rename_dict)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    for i, name in enumerate(METRIC_COLUMNS):
        out[name] = pd.to_numeric(out[f"v{i}"], errors="coerce")
    
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
    if ts.dt.tz is None:
        return ts.dt.tz_localize(tz)
    return ts.dt.tz_convert(tz)

def normalize_hour_key(df: pd.DataFrame, tz="Asia/Bangkok"):
    df = df.copy()
    if "timestamp" in df.columns:
        df["hour_key"] = safe_tz(df["timestamp"], tz=tz).dt.floor("H")
    return df

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp, tz: str = "Asia/Bangkok") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = normalize_hour_key(df, tz=tz)

    target_hour = safe_tz(pd.Series([at_ts]), tz=tz).dt.floor("H").iloc[0]
    
    y = x[x["hour_key"] == target_hour].copy()
    if y.empty:
        return pd.DataFrame()
    
    y = y.sort_values("timestamp")
    if "channel" in y.columns:
        return y.groupby("channel").tail(1).reset_index(drop=True)
    return y.tail(1).reset_index(drop=True)

def current_and_yesterday_snapshots(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.NaT
    cur_ts = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts, tz=tz)
    y_snap = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1), tz=tz)
    return cur_snap, y_snap, cur_ts.floor("H")

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty:
        return dict(Sales=0.0, Orders=0.0, Ads=0.0, SaleRO=np.nan, AdsRO_avg=np.nan)
    
    sales = float(snap["sales"].sum())
    orders = float(snap["orders"].sum())
    ads = float(snap["ads"].sum())
    
    sale_ro = (sales / ads) if ads != 0 else np.nan
    
    ads_ro_vals = snap["ads_ro_raw"]
    ads_ro_avg = float(ads_ro_vals[ads_ro_vals > 0].mean())
    
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pct_delta(curr, prev):
    if pd.isna(prev) or pd.isna(curr) or prev == 0:
        return None
    return (curr - prev) * 100.0 / prev

def hourly_latest(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["hour_key"] = safe_tz(d["timestamp"], tz=tz).dt.floor("H")
    d = d.sort_values("timestamp").groupby(["channel", "hour_key"]).tail(1)
    d["day"] = d["hour_key"].dt.date
    d["hstr"] = d["hour_key"].dt.strftime("%H:%M")
    return d.reset_index(drop=True)

def calculate_hourly_values(df: pd.DataFrame, metric: str, tz="Asia/Bangkok"):
    if df.empty:
        return pd.DataFrame(columns=df.columns.tolist() + ["hour_key", "day", "hstr", "_val"])

    H = hourly_latest(df, tz=tz).sort_values(["channel", "hour_key"])
    
    diff_func = lambda s: s.diff().clip(lower=0)

    if metric in ("sales", "orders", "ads"):
        diff_col = H.groupby("channel")[metric].transform(diff_func)
        H["_val"] = diff_col.fillna(0.0)
    elif metric == "sale_day":
        H["_val"] = H["sales"].fillna(0.0)
    elif metric == "sale_ro":
        ds = H.groupby("channel")["sales"].transform(diff_func).fillna(0.0)
        da = H.groupby("channel")["ads"].transform(diff_func).fillna(0.0).replace(0, np.nan)
        ro = ds / da
        ro = ro.replace([np.inf, -np.inf], np.nan).clip(upper=50)
        H["_val"] = ro.fillna(0.0)
    elif metric == "ads_ro":
        H['sales_from_ads'] = H['ads'] * H['ads_ro_raw']
        ds_from_ads = H.groupby("channel")['sales_from_ads'].transform(diff_func).fillna(0.0)
        da = H.groupby("channel")["ads"].transform(diff_func).fillna(0.0).replace(0, np.nan)
        ro = ds_from_ads / da
        ro = ro.replace([np.inf, -np.inf], np.nan).clip(upper=50)
        H["_val"] = ro.fillna(0.0)
    else:
        H["_val"] = 0.0
    
    H["_val"] = pd.to_numeric(H["_val"], errors='coerce').fillna(0.0)
    return H

def build_overlay_by_day(df: pd.DataFrame, metric: str, tz="Asia/Bangkok"):
    H_with_vals = calculate_hourly_values(df, metric, tz)
    if H_with_vals.empty:
        return pd.DataFrame()
    
    agg_function = "mean" if metric in ("sale_ro", "ads_ro") else "sum"
    
    pivot = H_with_vals.pivot_table(index="hstr", columns="day", values="_val", aggfunc=agg_function).sort_index()
    return pivot

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def display_kpi_metrics(cur: dict, prev: dict, snapshot_hour):
    cols = st.columns(5)
    cols[0].metric("Sales", f"{cur['Sales']:,.0f}",
                   delta=(f"{pct_delta(cur['Sales'], prev['Sales']):+.1f}%" if prev['Sales'] else None))
    cols[1].metric("Orders", f"{cur['Orders']:,.0f}",
                   delta=(f"{pct_delta(cur['Orders'], prev['Orders']):+.1f}%" if prev['Orders'] else None))
    cols[2].metric("Budget (Ads)", f"{cur['Ads']:,.0f}",
                   delta=(f"{pct_delta(cur['Ads'], prev['Ads']):+.1f}%" if prev['Ads'] else None))
    cols[3].metric("SaleRO (Cumulative)",
                   f"{cur['SaleRO']:.2f}" if not pd.isna(cur["SaleRO"]) else "-",
                   delta=(f"{pct_delta(cur['SaleRO'], prev['SaleRO']):+.1f}%" if not pd.isna(prev["SaleRO"]) else None))
    cols[4].metric("AdsRO (Snapshot Avg)",
                   f"{cur['AdsRO_avg']:.2f}" if not pd.isna(cur["AdsRO_avg"]) else "-",
                   delta=(f"{pct_delta(cur['AdsRO_avg'], prev['AdsRO_avg']):+.1f}%" if not pd.isna(prev["AdsRO_avg"]) else None))
    if pd.notna(snapshot_hour):
        st.caption(f"Comparing snapshot at {snapshot_hour.strftime('%Y-%m-%d %H:00')}")

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.sidebar.header("Filters")
    if st.sidebar.button("Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

    try:
        wide = load_wide_df()
        df_long = build_long(wide)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.error(f"Details: {e}")
        st.stop()

    tz = "Asia/Bangkok"
    now_ts = pd.Timestamp.now(tz=tz)

    if df_long.empty:
        st.warning("No data found after processing. The source may be empty or in an invalid format.")
        st.stop()
        
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
    if not isinstance(d1, date_type) or not isinstance(d2, date_type):
        st.warning("Please select a valid date range.")
        st.stop()
        
    start_ts = pd.Timestamp.combine(d1, pd.Timestamp.min.time()).tz_localize(tz)
    end_ts = pd.Timestamp.combine(d2, pd.Timestamp.max.time()).tz_localize(tz)

    all_channels = sorted(df_long["channel"].dropna().unique().tolist())
    
    # --- NEW: Channel Grouping Logic ---
    jen_channels = [ch for ch in STAFF_CHANNELS["jen"] if ch in all_channels]
    fon_channels = [ch for ch in STAFF_CHANNELS["fon"] if ch in all_channels]
    
    jen_fon_channels = set(jen_channels + fon_channels)
    mint_channels = [ch for ch in all_channels if ch not in jen_fon_channels]

    # Create options with groups
    chan_options = ["[All]", "[Jen]", "[Fon]", "[Mint]"] + all_channels
    chosen = st.sidebar.multiselect("Channels", options=chan_options, default=["[All]"])

    selected_channels = []
    if not chosen:
        selected_channels = []
    elif "[All]" in chosen:
        selected_channels = all_channels
    else:
        temp_channels = set()
        if "[Jen]" in chosen:
            temp_channels.update(jen_channels)
        if "[Fon]" in chosen:
            temp_channels.update(fon_channels)
        if "[Mint]" in chosen:
            temp_channels.update(mint_channels)
        
        # Add individual channels that are not part of a selected group
        for ch in chosen:
            if ch not in ["[Jen]", "[Fon]", "[Mint]"]:
                temp_channels.add(ch)
        selected_channels = sorted(list(temp_channels))


    page = st.sidebar.radio("Page", ["Overview", "Channel", "Compare"])
    
    # --- NEW: Dynamic Title ---
    first_time_col_name = ""
    for i, col in enumerate(wide.columns):
        if is_time_col(str(col)):
            first_time_col_name = str(col)
            break
            
    st.title(f"Shopee ROAS Dashboard {first_time_col_name}")
    st.caption(f"Last refresh: {now_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    mask = (
        (df_long["timestamp"] >= start_ts) &
        (df_long["timestamp"] <= end_ts) &
        (df_long["channel"].isin(selected_channels))
    )
    d_filtered = df_long.loc[mask].copy()

    if page == "Overview":
        st.subheader("Overview (All selected channels)")
        if d_filtered.empty:
            st.warning("No data in selected period for the chosen channels.")
            st.stop()

        cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d_filtered, tz=tz)
        cur_kpis = kpis_from_snapshot(cur_snap)
        prev_kpis = kpis_from_snapshot(y_snap)
        display_kpi_metrics(cur_kpis, prev_kpis, cur_hour)

        st.markdown("### Hourly Performance Overlay")
        metric_options = ["sales", "orders", "ads", "sale_ro", "ads_ro", "sale_day", "ads_ro & sale_ro"]
        metric = st.selectbox("Metric to plot:", options=metric_options, index=0)

        show_heatmap = True
        piv_for_heatmap = None

        if metric == "ads_ro & sale_ro":
            show_heatmap = False
            piv_sale = build_overlay_by_day(d_filtered, "sale_ro", tz=tz)
            piv_ads = build_overlay_by_day(d_filtered, "ads_ro", tz=tz)

            fig = go.Figure()
            if not piv_sale.empty:
                for day in piv_sale.columns:
                    fig.add_trace(go.Scatter(x=piv_sale.index, y=piv_sale[day], mode="lines+markers", 
                                              name=f"{str(day)} (Sale RO)", line=dict(dash='solid')))
            if not piv_ads.empty:
                for day in piv_ads.columns:
                    fig.add_trace(go.Scatter(x=piv_ads.index, y=piv_ads[day], mode="lines+markers", 
                                              name=f"{str(day)} (Ads RO)", line=dict(dash='dash')))
            
            if not piv_sale.empty or not piv_ads.empty:
                fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title="ROAS")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data to plot for ROAS comparison.")
        
        else:
            piv = build_overlay_by_day(d_filtered, metric, tz=tz)
            piv_for_heatmap = piv
            if piv.empty:
                st.info("No data to plot for the selected metric.")
            else:
                fig = go.Figure()
                for day in piv.columns:
                    fig.add_trace(go.Scatter(x=piv.index, y=piv[day], mode="lines+markers", name=str(day)))
                title = "Cumulative Sales" if metric == "sale_day" else f"Hourly {metric.replace('_', ' ').title()}"
                fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=title)
                st.plotly_chart(fig, use_container_width=True)

        if show_heatmap:
            st.markdown("### Prime Hours Heatmap")
            if piv_for_heatmap is not None and not piv_for_heatmap.empty:
                fig_hm = px.imshow(
                    piv_for_heatmap.T,
                    aspect="auto",
                    labels=dict(x="Hour", y="Day", color=f"Hourly {metric}"),
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("No data for heatmap.")
        
        st.markdown("### Advertising Credits Are Low")
        
        credit_threshold = st.selectbox(
            "Show channels with credits below:",
            options=[500, 1000, 1500, 2000, 3000],
            index=0
        )

        latest_snapshot_all = df_long[df_long['channel'].isin(selected_channels)].sort_values('timestamp').groupby('channel').tail(1).reset_index()

        low_credit_channels = latest_snapshot_all[
            (latest_snapshot_all['misc'].notna()) &
            (latest_snapshot_all['misc'] < credit_threshold) &
            (latest_snapshot_all['ads'].notna()) &
            (latest_snapshot_all['ads'] > 0)
        ].copy()

        if low_credit_channels.empty:
            st.info(f"No active channels found with credits below {credit_threshold:,.0f}.")
        else:
            num_channels = len(low_credit_channels)
            num_cols = 6
            
            low_credit_channels = low_credit_channels.sort_values('misc')

            for i in range(0, num_channels, num_cols):
                cols = st.columns(num_cols)
                row_data = low_credit_channels.iloc[i : i + num_cols]
                
                for idx, col in enumerate(cols):
                    if idx < len(row_data):
                        channel_info = row_data.iloc[idx]
                        with col:
                            st.markdown(f"**{channel_info['channel']}**")
                            st.markdown(f"เครดิตคงเหลือ: **{channel_info['misc']:,.0f}**")
                            st.markdown(f"สถานะแอด: **เปิดใช้งาน**")
                            st.markdown("---")
        
        # --- CORRECTED: Campaign Data Display Section ---
        st.markdown("### Campaign Data")
        
        view_mode = st.radio(
            "เลือกมุมมอง:",
            ("แสดงรายละเอียด (Formatted)", "แสดงข้อมูลดิบ (Raw)"),
            horizontal=True,
            label_visibility="collapsed"
        )

        # Logic to dynamically find the first two columns (channel and campaign data)
        first_time_col_index = -1
        for i, col in enumerate(wide.columns):
            if is_time_col(str(col)):
                first_time_col_index = i
                break
        
        # Check if we found at least two ID columns before the time columns
        if first_time_col_index == -1 or first_time_col_index < 2:
            st.warning("ไม่สามารถระบุคอลัมน์ channel และ campaign จากข้อมูลดิบได้")
        else:
            id_cols = wide.columns[:first_time_col_index]
            channel_col_name = id_cols[0]
            campaign_col_name = id_cols[1]

            campaign_data_df = wide[[channel_col_name, campaign_col_name]].copy()
            campaign_data_df.columns = ['channel', 'campaign_data_string']

            if view_mode == "แสดงข้อมูลดิบ (Raw)":
                st.table(campaign_data_df[campaign_data_df['channel'].isin(selected_channels)])
            
            elif view_mode == "แสดงรายละเอียด (Formatted)":
                # Filter data based on selected channels BEFORE processing
                df_long_filtered_for_rank = df_long[df_long['channel'].isin(selected_channels)]
                campaign_data_df_filtered = campaign_data_df[campaign_data_df['channel'].isin(selected_channels)]
                
                if df_long_filtered_for_rank.empty:
                    st.info("ไม่พบข้อมูลสำหรับร้านค้าที่เลือก")
                    st.stop()
                    
                # Prepare latest daily stats for merging
                latest_daily_stats = df_long_filtered_for_rank.loc[df_long_filtered_for_rank.groupby('channel')['timestamp'].idxmax()].copy()
                latest_daily_stats['sale_ro_day'] = latest_daily_stats['sales'] / latest_daily_stats['ads'].replace(0, np.nan)
                latest_daily_stats.rename(columns={'ads_ro_raw': 'ads_ro_day'}, inplace=True)
                
                # --- Calculate Today's Ranks ---
                ranking_data = latest_daily_stats[['channel', 'sales']].copy()
                ranking_data.rename(columns={'sales': 'sale_day'}, inplace=True)
                ranking_data['sale_day'].fillna(0, inplace=True)
                ranking_data['rank_sale'] = ranking_data['sale_day'].rank(method='dense', ascending=False).astype(int)
                rank_dict = pd.Series(ranking_data.rank_sale.values, index=ranking_data.channel).to_dict()

                # --- NEW: Calculate Yesterday's Ranks & Growth Rank ---
                yesterday_ts = df_long_filtered_for_rank['timestamp'].max() - pd.Timedelta(days=1)
                yesterday_stats = pick_snapshot_at(df_long_filtered_for_rank, yesterday_ts, tz=tz)
                
                # Merge today's and yesterday's sales data
                comparison_df = ranking_data[['channel', 'sale_day']].copy()
                if not yesterday_stats.empty:
                    last_sales_df = yesterday_stats[['channel', 'sales']].copy()
                    last_sales_df.rename(columns={'sales': 'last_sale_day'}, inplace=True)
                    comparison_df = pd.merge(comparison_df, last_sales_df, on='channel', how='left')
                else:
                    comparison_df['last_sale_day'] = np.nan
                
                comparison_df.fillna(0, inplace=True)
                comparison_df['growth'] = comparison_df['sale_day'] - comparison_df['last_sale_day']
                comparison_df['rank_last_sale'] = comparison_df['growth'].rank(method='dense', ascending=False).astype(int)
                rank_last_dict = pd.Series(comparison_df.rank_last_sale.values, index=comparison_df.channel).to_dict()


                all_rows_to_display = []
                
                # --- Iterate in ORIGINAL order ---
                unique_channels = campaign_data_df_filtered['channel'].unique()
                
                channel_num = 0
                for channel_name in unique_channels:
                    channel_num += 1
                    is_first_row_for_channel = True
                    
                    details_string = campaign_data_df_filtered[campaign_data_df_filtered['channel'] == channel_name]['campaign_data_string'].iloc[0]
                    
                    try:
                        data_dict = ast.literal_eval(details_string)
                        setting_info = data_dict.get('setting', {})
                        parsed_campaigns = data_dict.get('campaigns', [])
                    except (ValueError, SyntaxError):
                        setting_info = {}
                        parsed_campaigns = []
                    
                    # Get daily stats for the current channel
                    channel_stats = latest_daily_stats[latest_daily_stats['channel'] == channel_name]
                    sale_ro_day_val = channel_stats['sale_ro_day'].iloc[0] if not channel_stats.empty else np.nan
                    ads_ro_day_val = channel_stats['ads_ro_day'].iloc[0] if not channel_stats.empty else np.nan
                    sale_day_val = channel_stats['sales'].iloc[0] if not channel_stats.empty else np.nan
                    saleads_day_val = channel_stats['view'].iloc[0] if not channel_stats.empty else np.nan
                    
                    # Get ranks from dictionaries
                    channel_rank = rank_dict.get(channel_name, '')
                    channel_last_rank = rank_last_dict.get(channel_name, '')

                    # Get yesterday's sales for the new column
                    last_sale_day_val = comparison_df[comparison_df['channel'] == channel_name]['last_sale_day'].iloc[0] if channel_name in comparison_df['channel'].values else np.nan


                    if not parsed_campaigns:
                        row_data = {
                            'No.': str(channel_num),
                            'channel': channel_name,
                            'type': setting_info.get('type', ''),
                            'GMV_Q': setting_info.get('gmv_quota'),
                            'GMV_U': setting_info.get('gmv_user'),
                            'AUTO_Q': setting_info.get('auto_quota'),
                            'AUTO_U': setting_info.get('auto_user'),
                            'id': '',
                            'budget': np.nan,
                            'sales': np.nan,
                            'orders': np.nan,
                            'roas': np.nan,
                            'SaleRO (Day)': sale_ro_day_val,
                            'AdsRO (Day)': ads_ro_day_val,
                            'saleads_day': saleads_day_val,
                            'sale_day': sale_day_val,
                            'salelast_day': last_sale_day_val,
                            'rank_sale': str(channel_rank),
                            'rank_last_sale': str(channel_last_rank),
                        }
                        all_rows_to_display.append(row_data)
                    else:
                        for campaign in parsed_campaigns:
                            row_data = {
                                'No.': str(channel_num) if is_first_row_for_channel else '',
                                'channel': channel_name if is_first_row_for_channel else '',
                                'type': setting_info.get('type') if is_first_row_for_channel else '',
                                'GMV_Q': setting_info.get('gmv_quota') if is_first_row_for_channel else np.nan,
                                'GMV_U': setting_info.get('gmv_user') if is_first_row_for_channel else np.nan,
                                'AUTO_Q': setting_info.get('auto_quota') if is_first_row_for_channel else np.nan,
                                'AUTO_U': setting_info.get('auto_user') if is_first_row_for_channel else np.nan,
                                'id': campaign.get('id'),
                                'budget': campaign.get('budget'),
                                'sales': campaign.get('sales'),
                                'orders': campaign.get('orders'),
                                'roas': campaign.get('roas'),
                                'SaleRO (Day)': sale_ro_day_val if is_first_row_for_channel else np.nan,
                                'AdsRO (Day)': ads_ro_day_val if is_first_row_for_channel else np.nan,
                                'saleads_day': saleads_day_val if is_first_row_for_channel else np.nan,
                                'sale_day': sale_day_val if is_first_row_for_channel else np.nan,
                                'salelast_day': last_sale_day_val if is_first_row_for_channel else np.nan,
                                'rank_sale': str(channel_rank) if is_first_row_for_channel else '',
                                'rank_last_sale': str(channel_last_rank) if is_first_row_for_channel else '',
                            }
                            all_rows_to_display.append(row_data)
                            is_first_row_for_channel = False
                
                if not all_rows_to_display:
                    st.info("ไม่พบข้อมูลแคมเปญที่สามารถจัดรูปแบบได้")
                else:
                    display_df = pd.DataFrame(all_rows_to_display)
                    
                    # Calculate height for dataframe to avoid scrollbar
                    height = (len(display_df) + 1) * 35 + 3

                    # Define formatters for styling
                    formatters = {
                        'budget': '{:,.0f}',
                        'sales': '{:,.0f}',
                        'orders': '{:,.0f}',
                        'roas': '{:.2f}',
                        'SaleRO (Day)': '{:.2f}',
                        'AdsRO (Day)': '{:.2f}',
                        'GMV_Q': '{:.1f}',
                        'GMV_U': '{:.0f}',
                        'AUTO_Q': '{:.1f}',
                        'AUTO_U': '{:.0f}',
                        'sale_day': '{:,.0f}',
                        'saleads_day': '{:,.0f}',
                        'salelast_day': '{:,.0f}',
                    }

                    st.dataframe(
                        display_df.style.format(formatters, na_rep=''),
                        use_container_width=True,
                        height=height,
                        hide_index=True
                    )

    elif page == "Channel":
        st.header("เจาะลึกรายร้านค้า (Channel Detail)")
        
        ch = st.selectbox("เลือกร้านค้าเพื่อเจาะลึก", options=all_channels, key="channel_select")
        
        ch_df = d_filtered[d_filtered["channel"] == ch].copy()
        
        st.subheader(f"สรุปสำหรับร้าน: `{ch}`")

        if ch_df.empty:
            st.warning("ไม่พบข้อมูลสำหรับร้านค้านี้ในช่วงเวลาที่เลือก")
            st.stop()

        cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(ch_df, tz=tz)
        cur_kpis = kpis_from_snapshot(cur_snap)
        prev_kpis = kpis_from_snapshot(y_snap)
        display_kpi_metrics(cur_kpis, prev_kpis, cur_hour)

        st.markdown(f"### Hourly Performance Overlay for {ch}")
        metric_options_ch = ["sales", "orders", "ads", "sale_ro", "ads_ro", "sale_day", "ads_ro & sale_ro"]
        metric_ch = st.selectbox("Metric to plot:", options=metric_options_ch, index=0, key="channel_metric")

        show_heatmap_ch = True
        piv_for_heatmap_ch = None

        if metric_ch == "ads_ro & sale_ro":
            show_heatmap_ch = False
            piv_sale_ch = build_overlay_by_day(ch_df, "sale_ro", tz=tz)
            piv_ads_ch = build_overlay_by_day(ch_df, "ads_ro", tz=tz)
            
            fig_ch = go.Figure()
            if not piv_sale_ch.empty:
                for day in piv_sale_ch.columns:
                    fig_ch.add_trace(go.Scatter(x=piv_sale_ch.index, y=piv_sale_ch[day], mode="lines+markers", 
                                                  name=f"{str(day)} (Sale RO)", line=dict(dash='solid')))
            if not piv_ads_ch.empty:
                for day in piv_ads_ch.columns:
                    fig_ch.add_trace(go.Scatter(x=piv_ads_ch.index, y=piv_ads_ch[day], mode="lines+markers", 
                                                  name=f"{str(day)} (Ads RO)", line=dict(dash='dash')))
            
            if not piv_sale_ch.empty or not piv_ads_ch.empty:
                fig_ch.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title="ROAS")
                st.plotly_chart(fig_ch, use_container_width=True)
            else:
                st.info("No data to plot for ROAS comparison.")

        else:
            piv_ch = build_overlay_by_day(ch_df, metric_ch, tz=tz)
            piv_for_heatmap_ch = piv_ch
            if piv_ch.empty:
                st.info("No data to plot.")
            else:
                fig_ch = go.Figure()
                for day in piv_ch.columns:
                    fig_ch.add_trace(go.Scatter(x=piv_ch.index, y=piv_ch[day], mode="lines+markers", name=str(day)))
                title_ch = "Cumulative Sales" if metric_ch == "sale_day" else f"Hourly {metric_ch.replace('_', ' ').title()}"
                fig_ch.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=title_ch)
                st.plotly_chart(fig_ch, use_container_width=True)
                
        if show_heatmap_ch:
            st.markdown(f"### Prime Hours Heatmap for {ch}")
            if piv_for_heatmap_ch is not None and not piv_for_heatmap_ch.empty:
                fig_hm_ch = px.imshow(
                    piv_for_heatmap_ch.T,
                    aspect="auto",
                    labels=dict(x="Hour", y="Day", color=f"Hourly {metric_ch}"),
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_hm_ch, use_container_width=True)
            else:
                st.info("No data for heatmap.")
        
        # --- NEW: Campaign Data Table for Channel Page ---
        st.markdown("### Campaign Data")
        
        # Filter wide data for the selected channel
        campaign_data_df_channel = wide[wide.iloc[:, 0] == ch]

        # Reuse the logic from Overview page, but with channel-specific data
        # Logic to dynamically find the first two columns
        first_time_col_index = -1
        for i, col in enumerate(wide.columns):
            if is_time_col(str(col)):
                first_time_col_index = i
                break
        
        if first_time_col_index == -1 or first_time_col_index < 2:
            st.warning("ไม่สามารถระบุคอลัมน์ channel และ campaign จากข้อมูลดิบได้")
        else:
            id_cols = wide.columns[:first_time_col_index]
            channel_col_name = id_cols[0]
            campaign_col_name = id_cols[1]

            campaign_data_df = wide[[channel_col_name, campaign_col_name]].copy()
            campaign_data_df.columns = ['channel', 'campaign_data_string']
            campaign_data_df_filtered = campaign_data_df[campaign_data_df['channel'] == ch]

            df_long_filtered_for_rank = df_long[df_long['channel'] == ch]
            if not df_long_filtered_for_rank.empty:
                latest_daily_stats = df_long_filtered_for_rank.loc[df_long_filtered_for_rank.groupby('channel')['timestamp'].idxmax()].copy()
                latest_daily_stats['sale_ro_day'] = latest_daily_stats['sales'] / latest_daily_stats['ads'].replace(0, np.nan)
                latest_daily_stats.rename(columns={'ads_ro_raw': 'ads_ro_day'}, inplace=True)
                
                all_rows_to_display = []
                channel_num = 0
                for channel_name in campaign_data_df_filtered['channel'].unique():
                    channel_num += 1
                    # ... (rest of the table generation logic)
                    is_first_row_for_channel = True
                    details_string = campaign_data_df_filtered[campaign_data_df_filtered['channel'] == channel_name]['campaign_data_string'].iloc[0]
                    try:
                        data_dict = ast.literal_eval(details_string)
                        setting_info = data_dict.get('setting', {})
                        parsed_campaigns = data_dict.get('campaigns', [])
                    except (ValueError, SyntaxError):
                        setting_info = {}
                        parsed_campaigns = []
                    
                    channel_stats = latest_daily_stats[latest_daily_stats['channel'] == channel_name]
                    # ... (get all other stats like sale_day_val, etc.)
                    sale_ro_day_val = channel_stats['sale_ro_day'].iloc[0] if not channel_stats.empty else np.nan
                    ads_ro_day_val = channel_stats['ads_ro_day'].iloc[0] if not channel_stats.empty else np.nan
                    sale_day_val = channel_stats['sales'].iloc[0] if not channel_stats.empty else np.nan
                    saleads_day_val = channel_stats['view'].iloc[0] if not channel_stats.empty else np.nan
                    
                    # For the channel page, ranks are not applicable, so we can hide them or show 'N/A'
                    if not parsed_campaigns:
                        row_data = {
                            'No.': str(channel_num),
                            'channel': channel_name,
                            'type': setting_info.get('type', ''),
                            'GMV_Q': setting_info.get('gmv_quota'),
                            'GMV_U': setting_info.get('gmv_user'),
                            'AUTO_Q': setting_info.get('auto_quota'),
                            'AUTO_U': setting_info.get('auto_user'),
                            'id': '',
                            'budget': np.nan,
                            'sales': np.nan,
                            'orders': np.nan,
                            'roas': np.nan,
                            'SaleRO (Day)': sale_ro_day_val,
                            'AdsRO (Day)': ads_ro_day_val,
                            'saleads_day': saleads_day_val,
                            'sale_day': sale_day_val,
                        }
                        all_rows_to_display.append(row_data)
                    else:
                        for campaign in parsed_campaigns:
                             row_data = {
                                'No.': str(channel_num) if is_first_row_for_channel else '',
                                'channel': channel_name if is_first_row_for_channel else '',
                                'type': setting_info.get('type') if is_first_row_for_channel else '',
                                'GMV_Q': setting_info.get('gmv_quota') if is_first_row_for_channel else np.nan,
                                'GMV_U': setting_info.get('gmv_user') if is_first_row_for_channel else np.nan,
                                'AUTO_Q': setting_info.get('auto_quota') if is_first_row_for_channel else np.nan,
                                'AUTO_U': setting_info.get('auto_user') if is_first_row_for_channel else np.nan,
                                'id': campaign.get('id'),
                                'budget': campaign.get('budget'),
                                'sales': campaign.get('sales'),
                                'orders': campaign.get('orders'),
                                'roas': campaign.get('roas'),
                                'SaleRO (Day)': sale_ro_day_val if is_first_row_for_channel else np.nan,
                                'AdsRO (Day)': ads_ro_day_val if is_first_row_for_channel else np.nan,
                                'saleads_day': saleads_day_val if is_first_row_for_channel else np.nan,
                                'sale_day': sale_day_val if is_first_row_for_channel else np.nan,
                            }
                             all_rows_to_display.append(row_data)
                             is_first_row_for_channel = False

                if all_rows_to_display:
                    display_df = pd.DataFrame(all_rows_to_display)
                    height = (len(display_df) + 1) * 35 + 3
                    formatters = {
                        'budget': '{:,.0f}','sales': '{:,.0f}','orders': '{:,.0f}','roas': '{:.2f}',
                        'SaleRO (Day)': '{:.2f}','AdsRO (Day)': '{:.2f}','GMV_Q': '{:.1f}',
                        'GMV_U': '{:.0f}','AUTO_Q': '{:.1f}','AUTO_U': '{:.0f}',
                        'sale_day': '{:,.0f}','saleads_day': '{:,.0f}',
                    }
                    st.dataframe(
                        display_df.style.format(formatters, na_rep=''),
                        use_container_width=True, height=height, hide_index=True
                    )


    elif page == "Compare":
        st.subheader("Channel Comparison")
        if len(all_channels) < 2:
            st.info("You need at least 2 channels in your data to use the compare feature.")
            st.stop()

        pick = st.multiselect("Pick 2–4 channels", options=all_channels, default=all_channels[:min(4, len(all_channels))], max_selections=4)
        if len(pick) < 2:
            st.info("Please pick at least 2 channels to compare.")
            st.stop()

        sub = d_filtered[d_filtered["channel"].isin(pick)].copy()
        if sub.empty:
            st.warning("No data for the selected channels in this period.")
            st.stop()

        st.markdown("#### Cumulative KPI Comparison")
        H = hourly_latest(sub, tz=tz)
        latest_snap = H.sort_values("hour_key").groupby("channel").tail(1)
        kpi_comparison = latest_snap.groupby("channel").agg(
            Total_Sales=("sales", "sum"),
            Total_Orders=("orders", "sum"),
            Total_Ads=("ads", "sum"),
            Avg_SaleRO=("SaleRO", "mean")
        ).reset_index()
        st.dataframe(kpi_comparison.round(2), use_container_width=True)

        st.markdown("#### Hourly Performance vs Baseline")
        base = st.selectbox("Baseline channel", options=pick, index=0)
        met_options = {
            "Sales (Hourly)": "sales", 
            "Orders (Hourly)": "orders", 
            "Ads (Hourly)": "ads", 
            "SaleRO (Hourly)": "sale_ro",
            "AdsRO (Hourly)": "ads_ro"
        }
        met_display = st.selectbox("Metric", options=list(met_options.keys()), index=0)
        met = met_options[met_display]

        H_with_vals = calculate_hourly_values(sub, met, tz=tz)
        if H_with_vals.empty:
            st.warning("No hourly data to compare for the selected metric.")
            st.stop()
            
        piv_compare = H_with_vals.pivot_table(index="hour_key", columns="channel", values="_val", aggfunc="sum").fillna(0)

        if base not in piv_compare.columns:
            st.warning(f"Baseline channel '{base}' has no data for this metric. Please choose another.")
            st.stop()
            
        base_series = piv_compare[base].replace(0, np.nan)
        rel = (piv_compare.div(base_series, axis=0) - 1.0) * 100.0

        fig = go.Figure()
        for c in rel.columns:
            if c == base: continue
            fig.add_trace(go.Scatter(x=rel.index.strftime("%Y-%m-%d %H:%M"), y=rel[c], name=f"{c} vs {base}", mode="lines"))
        
        fig.update_layout(height=420, xaxis_title="Hour", yaxis_title=f"% Difference in {met_display} vs {base}")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
