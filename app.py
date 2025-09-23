# app.py
# Shopee Campaign Monitor - Hybrid Dashboard
# ใช้โครงสร้าง UI แบบเก่า (Overview, Channel, Compare) กับข้อมูล Snapshot แบบใหม่
#
# pip: streamlit pandas

import io
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Campaign Monitor", layout="wide")

# -----------------------------------------------------------------------------
# Data Loading and Processing (จากเวอร์ชันใหม่)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Fetching latest data...")
def load_data_from_url(url):
    """
    ดึงข้อมูล CSV จาก URL และแปลงให้เป็น DataFrame ที่พร้อมใช้งาน
    """
    if not url:
        st.error("Missing Secrets: ROAS_CSV_URL is not set.")
        return pd.DataFrame()
    
    import requests
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        csv_text = response.text
        df_raw = pd.read_csv(io.StringIO(csv_text))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from URL: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read CSV data: {e}")
        return pd.DataFrame()

    if df_raw.shape[1] < 2:
        st.warning("Data does not have at least two columns (channel, json_data).")
        return pd.DataFrame()
        
    df_raw.columns = ['channel', 'json_data']
    
    all_campaigns = []
    
    for index, row in df_raw.iterrows():
        channel_name = row['channel']
        json_string = row['json_data']
        
        try:
            data = json.loads(json_string)
            setting_info = data.get('setting', {})
            status = data.get('status', 'unknown')
            campaigns = data.get('campaigns', [])
            
            if not campaigns:
                all_campaigns.append({
                    'channel': channel_name, 'status': status,
                    'setting_type': setting_info.get('type'), 'gmv_quota': setting_info.get('gmv_quota'),
                    'auto_quota': setting_info.get('auto_quota'), 'campaign_id': None,
                    'budget': 0, 'clicks': 0, 'orders': 0, 'sales': 0, 'roas': 0,
                })
            else:
                for campaign in campaigns:
                    all_campaigns.append({
                        'channel': channel_name, 'status': status,
                        'setting_type': setting_info.get('type'), 'gmv_quota': setting_info.get('gmv_quota'),
                        'auto_quota': setting_info.get('auto_quota'), 'campaign_id': campaign.get('id'),
                        'budget': campaign.get('budget', 0), 'clicks': campaign.get('clicks', 0),
                        'orders': campaign.get('orders', 0), 'sales': campaign.get('sales', 0),
                        'roas': campaign.get('roas', 0),
                    })
        except (json.JSONDecodeError, TypeError):
            all_campaigns.append({
                'channel': channel_name, 'status': 'parse_error',
                'setting_type': None, 'gmv_quota': None, 'auto_quota': None,
                'campaign_id': 'Error reading data', 'budget': 0, 'clicks': 0,
                'orders': 0, 'sales': 0, 'roas': 0,
            })
            
    if not all_campaigns:
        return pd.DataFrame()

    return pd.DataFrame(all_campaigns)

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------
def display_summary_metrics(df):
    """แสดง KPI สรุปภาพรวม"""
    total_budget = df['budget'].sum()
    total_sales = df['sales'].sum()
    overall_roas = (total_sales / total_budget) if total_budget > 0 else 0
    active_campaigns = len(df[df['status'] == 'active'].dropna(subset=['campaign_id']))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("งบประมาณรวม (Budget)", f"{total_budget:,.0f}")
    c2.metric("ยอดขายรวม (Sales)", f"{total_sales:,.0f}")
    c3.metric("ROAS เฉลี่ย", f"{overall_roas:,.2f}")
    c4.metric("จำนวนแคมเปญที่ทำงานอยู่", f"{active_campaigns}")

def display_campaign_table(df):
    """แสดงตารางข้อมูลแคมเปญแบบละเอียด"""
    df_display = df.copy()
    for col in ['budget', 'sales', 'orders', 'clicks']:
        df_display[col] = df_display[col].map('{:,.0f}'.format)
    df_display['roas'] = df_display['roas'].map('{:,.2f}'.format)
    
    display_cols = [
        'channel', 'campaign_id', 'status', 'setting_type', 
        'sales', 'budget', 'roas', 'orders', 'clicks',
        'gmv_quota', 'auto_quota'
    ]
    st.dataframe(df_display[display_cols], use_container_width=True)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main():
    st.title("📊 Shopee Campaign Monitor (Hybrid)")
    
    # --- Load Data ---
    csv_url = st.secrets.get("ROAS_CSV_URL", "")
    df = load_data_from_url(csv_url)

    if df.empty:
        st.warning("ไม่พบข้อมูลแคมเปญหลังจากประมวลผลแล้ว")
        st.stop()
        
    # --- Sidebar Filters & Page Navigation (จากเวอร์ชันเก่า) ---
    st.sidebar.header("⚙️ ควบคุมและตัวกรอง")
    page = st.sidebar.radio("เลือกหน้า", ["Overview", "Channel", "Compare"])
    st.sidebar.markdown("---")

    # Filters
    all_statuses = df['status'].unique()
    selected_status = st.sidebar.multiselect("สถานะ (Status)", options=all_statuses, default=[s for s in all_statuses if s != 'inactive'])
    
    all_settings = sorted(df['setting_type'].dropna().unique())
    selected_setting = st.sidebar.multiselect("ประเภท Setting", options=all_settings, default=all_settings)
    
    all_channels = sorted(df['channel'].unique())

    # --- Page Content ---
    if page == "Overview":
        st.header("ภาพรวมทั้งหมด (Overview)")
        selected_channels = st.sidebar.multiselect("ร้านค้า (Channel)", options=all_channels, default=all_channels)
        
        df_filtered = df[df['status'].isin(selected_status) & df['setting_type'].isin(selected_setting) & df['channel'].isin(selected_channels)]
        
        display_summary_metrics(df_filtered)
        st.markdown("---")
        st.subheader("ตารางข้อมูลทุกแคมเปญ")
        display_campaign_table(df_filtered)

    elif page == "Channel":
        st.header("เจาะลึกรายร้านค้า (Channel Detail)")
        
        if not all_channels:
            st.warning("ไม่พบข้อมูลร้านค้า")
            st.stop()
            
        # Channel selector for this page
        channel_to_view = st.sidebar.selectbox("เลือกร้านค้าเพื่อเจาะลึก", options=all_channels)
        
        df_filtered = df[df['channel'] == channel_to_view]
        
        st.subheader(f"สรุปสำหรับร้าน: `{channel_to_view}`")
        display_summary_metrics(df_filtered)
        st.markdown("---")
        st.subheader("แคมเปญทั้งหมดของร้านนี้")
        display_campaign_table(df_filtered)

    elif page == "Compare":
        st.header("เปรียบเทียบระหว่างร้านค้า (Compare Channels)")
        
        channels_to_compare = st.sidebar.multiselect(
            "เลือกร้านค้าเพื่อเปรียบเทียบ (2-5 ร้าน)",
            options=all_channels,
            default=all_channels[:min(len(all_channels), 4)]
        )

        if len(channels_to_compare) < 2:
            st.info("กรุณาเลือกอย่างน้อย 2 ร้านค้าเพื่อเปรียบเทียบ")
            st.stop()
            
        df_filtered = df[df['channel'].isin(channels_to_compare)]
        
        st.subheader("ตารางเปรียบเทียบผลสรุป")
        channel_summary = df_filtered.groupby('channel').agg(
            total_sales=('sales', 'sum'),
            total_budget=('budget', 'sum'),
            campaign_count=('campaign_id', 'nunique'),
        ).reset_index()
        
        channel_summary['roas'] = (channel_summary['total_sales'] / channel_summary['total_budget']).where(channel_summary['total_budget'] > 0, 0)
        
        st.dataframe(
            channel_summary.sort_values('total_sales', ascending=False).style.format({
                'total_sales': '{:,.0f}',
                'total_budget': '{:,.0f}',
                'roas': '{:,.2f}'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()

