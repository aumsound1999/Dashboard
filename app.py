# app.py
# Step 2: Extract and display active campaign names.

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
    NEW LOGIC for Step 2:
    This function now extracts the names (IDs) of the active campaigns.
    It uses the same reliable method of identifying a campaign (a list with len > 6)
    and then extracts the first element, which is the campaign ID.
    """
    if not isinstance(campaign_string, str):
        return ""

    try:
        campaign_list = ast.literal_eval(campaign_string)
        
        if not isinstance(campaign_list, list):
            return ""
            
        campaign_names = []
        for item in campaign_list:
            # A real campaign with performance data is a list with more than 6 elements.
            if isinstance(item, list) and len(item) > 6:
                # The first element is the campaign ID
                if isinstance(item[0], str):
                    campaign_names.append(item[0])
        
        # Join multiple campaign names with a newline character for better display
        return "\n".join(campaign_names)
    except (ValueError, SyntaxError):
        return ""

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.title("Step 2: Campaign Name Extraction")
    st.info("Now we are extracting the names of the active campaigns. If a channel has multiple campaigns, they will be displayed on separate lines.")

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
        
        # Apply the new extraction function
        summary_df['Active Campaign IDs'] = summary_df['Raw Campaign Data'].apply(extract_campaign_names)

        # --- Display the result ---
        st.markdown("### Active Campaign IDs per Channel")
        # Filter out channels with no active campaigns for a cleaner view
        active_channels_df = summary_df[summary_df['Active Campaign IDs'] != ""].copy()
        st.dataframe(
            active_channels_df[['Channel', 'Active Campaign IDs']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()

