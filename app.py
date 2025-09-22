# app.py
# Step 1: Correctly parse and count active campaigns.

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

def count_active_campaigns(campaign_string: str) -> int:
    """
    NEW ROBUST LOGIC:
    Parses the complex campaign string to count only the real, active campaigns.
    A real campaign is identified as a list containing detailed performance metrics (len > 6).
    This correctly ignores status indicators (e.g., ['gmvus2.0']) and plain text.
    """
    if not isinstance(campaign_string, str):
        return 0

    try:
        # Safely evaluate the string as a Python literal (e.g., "['a', ['b']]" -> ['a', ['b']])
        campaign_list = ast.literal_eval(campaign_string)
        
        # If it's not a list, it's not valid campaign data
        if not isinstance(campaign_list, list):
            return 0
            
        count = 0
        for item in campaign_list:
            # A real campaign with performance data is a list with more than 6 elements.
            if isinstance(item, list) and len(item) > 6:
                count += 1
        return count
    except (ValueError, SyntaxError):
        # This handles cases where the string is not a valid list format,
        # like "ไม่มีแอดที่เปิด" (No ads open).
        return 0

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.title("Step 1: Active Campaign Count Verification")
    st.info("This is a simplified view to verify the core logic. We are displaying the count of active campaigns for each channel based on the latest data.")

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
        # Select only the necessary columns
        summary_df = df_wide[[channel_col, campaign_col]].copy()
        summary_df.rename(columns={channel_col: 'Channel', campaign_col: 'Raw Campaign Data'}, inplace=True)
        
        # Apply the counting function to create the new column
        summary_df['Active Campaign Count'] = summary_df['Raw Campaign Data'].apply(count_active_campaigns)

        # --- Display the result ---
        st.markdown("### Active Campaign Count per Channel")
        st.dataframe(
            summary_df[['Channel', 'Active Campaign Count', 'Raw Campaign Data']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e) # Provides a full traceback for debugging

if __name__ == "__main__":
    main()

