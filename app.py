# app.py
# Final, fully functional version

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

# Constants for easier maintenance
METRIC_COLUMNS = ["sales", "orders", "ads", "view", "ads_ro_raw", "misc"]
V_COLUMNS = [f"v{i}" for i in range(len(METRIC_COLUMNS))]

# -----------------------------------------------------------------------------
# Helpers: detect & parse
# -----------------------------------------------------------------------------

TIME_COL_PATTERN = re.compile(r"^[A-Z]\d{1,2}\s+\d{1,2}:\d{1,2}$")

def is_time_col(col: str) -> bool:
    return isinstance(col, str) and TIME_COL_PATTERN.match(col.strip()) is not None

def parse_timestamps_from_headers(headers: list[str], tz: str = "Asia/Bangkok") -> dict:
    timestamps = {}
    now = pd.Timestamp.now(tz=tz)
    time_cols_only = [h for h in headers if is_time_col(h)]
    if not time_cols_only:
        return {h: pd.NaT for h in headers}
    temp_timestamps = {}
    previous_ts = pd.NaT
    for hdr in time_cols_only:
        hdr_strip = hdr.strip()
        m = re.match(r"^[A-Z](\d{1,2})\s+(\d{1,2}):(\d{1,2})$", hdr_strip)
        if not m: continue
        d, hh, mm = map(int, m.groups())
        if pd.isna(previous_ts):
            anchor_date_candidate = now
            for _ in range(45):
                if anchor_date_candidate.day == d: break
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
            try:
                ts = pd.Timestamp(year=previous_ts.year, month=previous_ts.month, day=d, hour=hh, minute=mm, tz=tz)
                if ts >= previous_ts:
                    ts -= pd.DateOffset(months=1)
                temp_timestamps[hdr] = ts
                previous_ts = ts
            except ValueError:
                try:
                    prev_month_date = previous_ts - pd.DateOffset(months=1)
                    ts = pd.Timestamp(year=prev_month_date.year, month=prev_month_date.month, day=d, hour=hh, minute=mm, tz=tz)
                    temp_timestamps[hdr] = ts
                    previous_ts = ts
                except ValueError:
                    temp_timestamps[hdr] = pd.NaT
    for hdr in headers:
        timestamps[hdr] = temp_timestamps.get(hdr, pd.NaT)
    return timestamps

def parse_metrics_cell(s: str):
    if not isinstance(s, str): return [np.nan] * 6
    s_clean = re.sub(r"[^0-9\.\-,]", "", s)
    parts = s_clean.split(",")
    nums = []
    for p in parts[:6]:
        if p.strip() == "":
            nums.append(np.nan)
            continue
        try:
            nums.append(float(p))
        except (ValueError, TypeError):
            nums.append(np.nan)
    while len(nums) < 6:
        nums.append(np.nan)
    return nums

def extract_campaign_details(campaign_string: str) -> list:
    """
    FINAL ROBUST LOGIC v2:
    Extracts the full performance data for each real campaign.
    A real campaign is a list with at least 7 elements (ID + 6 metrics).
    """
    if not isinstance(campaign_string, str):
        return []

    try:
        campaign_list = ast.literal_eval(campaign_string)
        parsed_data = []
        if not isinstance(campaign_list, list):
            return []
        
        for item in campaign_list:
            if isinstance(item, list) and len(item) > 6:
                try:
                    details = {
                        "Campaign ID": item[0],
                        "Budget": item[1],
                        "Ads Spent": item[2], # เงินที่ใช้
                        "Orders": item[3],
                        "View": item[4],
                        "Sales": item[5],
                        "ROAS": item[6],
                    }
                    parsed_data.append(details)
                except (IndexError, TypeError):
                    continue
        return parsed_data
    except (ValueError, SyntaxError):
        return []

# -----------------------------------------------------------------------------
# Data Loading & Transformation
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Fetching latest data...")
def fetch_csv_text():
    url = st.secrets.get("ROAS_CSV_URL", "")
    if not url: raise RuntimeError("Missing Secrets: ROAS_CSV_URL")
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=600, show_spinner=False)
def load_wide_df():
    return pd.read_csv(io.StringIO(fetch_csv_text()))

def long_from_wide(df_wide: pd.DataFrame, tz="Asia/Bangkok") -> pd.DataFrame:
    first_time_col_index = -1
    for i, col in enumerate(df_wide.columns):
        if is_time_col(str(col)):
            first_time_col_index = i
            break
    if first_time_col_index == -1: raise ValueError("No time columns detected.")
    id_cols = df_wide.columns[:first_time_col_index].tolist()
    time_cols = [col for col in df_wide.columns[first_time_col_index:] if is_time_col(str(col))]
    if not id_cols: raise ValueError("No identifier columns found.")
    ts_map = parse_timestamps_from_headers(df_wide.columns, tz=tz)
    m = df_wide.melt(id_vars=id_cols, value_vars=time_cols, var_name="time_col", value_name="raw")
    m["timestamp"] = m["time_col"].map(ts_map)
    V = pd.DataFrame(m["raw"].apply(parse_metrics_cell).tolist(), columns=V_COLUMNS)
    rename_dict = {id_cols[0]: 'channel'}
    if len(id_cols) > 1: rename_dict[id_cols[1]] = 'campaign_raw'
    out = pd.concat([m[["timestamp"] + id_cols], V], axis=1).rename(columns=rename_dict)
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    for i, name in enumerate(METRIC_COLUMNS):
        out[name] = pd.to_numeric(out[f"v{i}"], errors="coerce")
    out["SaleRO"] = out["sales"] / out["ads"].replace(0, np.nan)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def build_long(wide):
    return long_from_wide(wide)

def safe_tz(ts: pd.Series, tz="Asia/Bangkok"):
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.dt.tz is None: return ts.dt.tz_localize(tz)
    return ts.dt.tz_convert(tz)

def hourly_latest(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty: return pd.DataFrame()
    d = df.copy()
    d["hour_key"] = safe_tz(d["timestamp"], tz=tz).dt.floor("H")
    d = d.sort_values("timestamp").groupby(["channel", "hour_key"]).tail(1)
    d["day"] = d["hour_key"].dt.date
    d["hstr"] = d["hour_key"].dt.strftime("%H:%M")
    return d.reset_index(drop=True)

def calculate_hourly_values(df: pd.DataFrame, metric: str, tz="Asia/Bangkok"):
    if df.empty: return pd.DataFrame(columns=df.columns.tolist() + ["_val"])
    H = hourly_latest(df, tz=tz).sort_values(["channel", "hour_key"])
    diff_func = lambda s: s.diff().clip(lower=0)
    if metric in ("sales", "orders", "ads"): H["_val"] = H.groupby("channel")[metric].transform(diff_func).fillna(0.0)
    elif metric == "sale_day": H["_val"] = H["sales"].fillna(0.0)
    elif metric == "sale_ro":
        ds = H.groupby("channel")["sales"].transform(diff_func).fillna(0.0)
        da = H.groupby("channel")["ads"].transform(diff_func).fillna(0.0).replace(0, np.nan)
        ro = (ds / da).replace([np.inf, -np.inf], np.nan).clip(upper=50)
        H["_val"] = ro.fillna(0.0)
    elif metric == "ads_ro":
        H['sales_from_ads'] = H['ads'] * H['ads_ro_raw']
        ds_from_ads = H.groupby("channel")['sales_from_ads'].transform(diff_func).fillna(0.0)
        da = H.groupby("channel")["ads"].transform(diff_func).fillna(0.0).replace(0, np.nan)
        ro = (ds_from_ads / da).replace([np.inf, -np.inf], np.nan).clip(upper=50)
        H["_val"] = ro.fillna(0.0)
    else: H["_val"] = 0.0
    H["_val"] = pd.to_numeric(H["_val"], errors='coerce').fillna(0.0)
    return H

def build_overlay_by_day(df: pd.DataFrame, metric: str, tz="Asia/Bangkok"):
    H_with_vals = calculate_hourly_values(df, metric, tz)
    if H_with_vals.empty: return pd.DataFrame()
    agg_function = "mean" if metric in ("sale_ro", "ads_ro") else "sum"
    return H_with_vals.pivot_table(index="hstr", columns="day", values="_val", aggfunc=agg_function).sort_index()

def kpis_from_snapshot(snap: pd.DataFrame):
    if snap.empty: return dict(Sales=0.0, Orders=0.0, Ads=0.0, SaleRO=np.nan, AdsRO_avg=np.nan)
    sales = float(snap["sales"].sum())
    orders = float(snap["orders"].sum())
    ads = float(snap["ads"].sum())
    sale_ro = (sales / ads) if ads != 0 else np.nan
    ads_ro_vals = snap["ads_ro_raw"]
    ads_ro_avg = float(ads_ro_vals[ads_ro_vals > 0].mean())
    return dict(Sales=sales, Orders=orders, Ads=ads, SaleRO=sale_ro, AdsRO_avg=ads_ro_avg)

def pick_snapshot_at(df: pd.DataFrame, at_ts: pd.Timestamp, tz: str = "Asia/Bangkok") -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    if "timestamp" in df.columns:
        df["hour_key"] = safe_tz(df["timestamp"], tz=tz).dt.floor("H")
    target_hour = safe_tz(pd.Series([at_ts]), tz=tz).dt.floor("H").iloc[0]
    y = df[df["hour_key"] == target_hour].copy()
    if y.empty: return pd.DataFrame()
    y = y.sort_values("timestamp")
    if "channel" in y.columns:
        return y.groupby("channel").tail(1).reset_index(drop=True)
    return y.tail(1).reset_index(drop=True)

def current_and_yesterday_snapshots(df: pd.DataFrame, tz="Asia/Bangkok"):
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.NaT
    cur_ts = df["timestamp"].max()
    cur_snap = pick_snapshot_at(df, cur_ts, tz=tz)
    y_snap = pick_snapshot_at(df, cur_ts - pd.Timedelta(days=1), tz=tz)
    return cur_snap, y_snap, cur_ts.floor("H")

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.title("Shopee ROAS Dashboard")

    try:
        wide = load_wide_df()
        df_long = build_long(wide)
    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
        st.stop()
        
    st.sidebar.header("Filters")
    if st.sidebar.button("Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

    min_ts = df_long["timestamp"].min()
    max_ts = df_long["timestamp"].max()
    d1, d2 = st.sidebar.date_input(
        "Date range",
        value=(max_ts - pd.Timedelta(days=2), max_ts),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )
    if not isinstance(d1, date_type) or not isinstance(d2, date_type): st.stop()
    start_ts = pd.Timestamp.combine(d1, pd.Timestamp.min.time()).tz_localize("Asia/Bangkok")
    end_ts = pd.Timestamp.combine(d2, pd.Timestamp.max.time()).tz_localize("Asia/Bangkok")
    
    all_channels = sorted(df_long["channel"].dropna().unique().tolist())
    chosen = st.sidebar.multiselect("Channels", options=["[All]"] + all_channels, default=["[All]"])
    selected_channels = all_channels if "[All]" in chosen or not chosen else chosen
    
    mask = (df_long["timestamp"].between(start_ts, end_ts)) & (df_long["channel"].isin(selected_channels))
    d_filtered = df_long.loc[mask].copy()

    st.caption(f"Last refresh: {pd.Timestamp.now(tz='Asia/Bangkok').strftime('%Y-%m-%d %H:%M:%S')}")
    if d_filtered.empty:
        st.warning("No data in selected period.")
        st.stop()
    
    cur_snap, y_snap, cur_hour = current_and_yesterday_snapshots(d_filtered, tz="Asia/Bangkok")
    cur_kpis = kpis_from_snapshot(cur_snap)
    prev_kpis = kpis_from_snapshot(y_snap)
    
    cols = st.columns(5)
    cols[0].metric("Sales", f"{cur_kpis['Sales']:,.0f}", delta=f"{(cur_kpis['Sales'] - prev_kpis['Sales']):,.0f}")
    cols[1].metric("Orders", f"{cur_kpis['Orders']:,.0f}", delta=f"{(cur_kpis['Orders'] - prev_kpis['Orders']):,.0f}")
    cols[2].metric("Ads", f"{cur_kpis['Ads']:,.0f}", delta=f"{(cur_kpis['Ads'] - prev_kpis['Ads']):,.0f}")
    cols[3].metric("SaleRO", f"{cur_kpis['SaleRO']:.2f}" if pd.notna(cur_kpis['SaleRO']) else "-", delta=f"{(cur_kpis['SaleRO'] - prev_kpis['SaleRO']):.2f}" if pd.notna(cur_kpis['SaleRO']) and pd.notna(prev_kpis['SaleRO']) else None)
    cols[4].metric("AdsRO Avg", f"{cur_kpis['AdsRO_avg']:.2f}" if pd.notna(cur_kpis['AdsRO_avg']) else "-", delta=f"{(cur_kpis['AdsRO_avg'] - prev_kpis['AdsRO_avg']):.2f}" if pd.notna(cur_kpis['AdsRO_avg']) and pd.notna(prev_kpis['AdsRO_avg']) else None)

    st.markdown("### Hourly Performance Overlay")
    metric = st.selectbox("Metric to plot:", options=["sales", "orders", "ads", "sale_ro", "ads_ro", "sale_day"])

    piv = build_overlay_by_day(d_filtered, metric, tz="Asia/Bangkok")
    if piv.empty:
        st.info("No data to plot for the selected metric.")
    else:
        fig = go.Figure()
        for day in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[day], mode="lines+markers", name=str(day)))
        title = "Cumulative Sales" if metric == "sale_day" else f"Hourly {metric.replace('_', ' ').title()}"
        fig.update_layout(height=420, xaxis_title="Time (HH:MM)", yaxis_title=title)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Campaign Performance Summary")
    latest_snapshot_all = df_long.sort_values('timestamp').groupby('channel').tail(1)
    
    if 'campaign_raw' not in latest_snapshot_all.columns:
        st.warning("'Campaign' column not found.")
    else:
        all_campaigns = []
        for _, row in latest_snapshot_all.iterrows():
            details = extract_campaign_details(row['campaign_raw'])
            for camp in details:
                camp['Channel'] = row['channel']
                all_campaigns.append(camp)

        if not all_campaigns:
            st.info("No active campaign data found in the latest snapshot.")
        else:
            campaign_df = pd.DataFrame(all_campaigns)
            cols_order = ['Channel', 'Campaign ID', 'Budget', 'Ads Spent', 'Orders', 'View', 'Sales', 'ROAS']
            campaign_df = campaign_df[cols_order]
            st.dataframe(campaign_df.sort_values("ROAS", ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()

