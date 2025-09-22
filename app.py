# app.py
# Step 2: Extract campaign names AND count them.

import io
import re
import ast
import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Fetching latest data...")
def fetch_csv_text():
    """Fetches the CSV data from the URL stored in Streamlit secrets."""
    url = st.secrets.get("ROAS_CSV_URL", "")
    if not url:
        raise RuntimeError("Missing Secrets: ROAS_CSV_URL is not set.")
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not fetch data from URL: {e}")

# -----------------------------------------------------------------------------
# Core Logic: Campaign Parsing
# -----------------------------------------------------------------------------

def extract_campaign_names(campaign_string: str) -> str:
    """
    Extracts the names (IDs) of the active campaigns using a robust regex.
    """
    if not isinstance(campaign_string, str):
        return ""
    
    # Regex to find a string inside single quotes that contains letters, numbers, and at least one underscore.
    campaign_names = re.findall(r"\'([a-zA-Z0-9_]+)\'", campaign_string)
    
    return "\n".join(campaign_names)

def count_active_campaigns(campaign_string: str) -> int:
    """
    Counts the number of active campaigns using the same robust regex.
    """
    if not isinstance(campaign_string, str):
        return 0
        
    campaign_names = re.findall(r"\'([a-zA-Z0-9_]+)\'", campaign_string)
    return len(campaign_names)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.title("Step 2: Campaign Name Extraction and Count")
    st.info("We are now extracting campaign names and also providing a count of active campaigns.")

    try:
        # Load the raw data from Google Sheets
        csv_text = fetch_csv_text()
        df_wide = pd.read_csv(io.StringIO(csv_text))

        # --- Data Validation ---
        if df_wide.shape[1] < 2:
            st.error("The data sheet must have at least 2 columns ('name' and 'campaign' data).")
            st.stop()
            
        channel_col = df_wide.columns[0]
        campaign_col = df_wide.columns[1]

        # --- Create the summary table ---
        summary_df = df_wide[[channel_col, campaign_col]].copy()
        summary_df.rename(columns={channel_col: 'Channel', campaign_col: 'Raw Campaign Data'}, inplace=True)
        
        # Apply both functions
        summary_df['Active Campaign IDs'] = summary_df['Raw Campaign Data'].apply(extract_campaign_names)
        summary_df['Active Campaign Count'] = summary_df['Raw Campaign Data'].apply(count_active_campaigns)

        # --- Display the result ---
        st.markdown("### Active Campaigns per Channel")
        # Filter out channels with no active campaigns for a cleaner view
        active_channels_df = summary_df[summary_df['Active Campaign Count'] > 0].copy()
        st.dataframe(
            active_channels_df[['Channel', 'Active Campaign Count', 'Active Campaign IDs']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()

