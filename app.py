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
    CRITICAL FIX: Using a more robust Regex method.
    The previous method `ast.literal_eval` was too strict and failed on minor
    text formatting inconsistencies. This new method searches for the PATTERN
    of a campaign (a list with at least 6 commas) which is more reliable.
    """
    if not isinstance(campaign_string, str):
        return 0
    
    # This regex finds patterns that look like a list `[...]` containing at least 6 commas.
    # This is the most reliable signature of a campaign with performance data.
    # `[^\[\]]*` matches any character except brackets, to contain the search within one list item.
    matches = re.findall(r"\[[^\[\]]*?,[^\[\]]*?,[^\[\]]*?,[^\[\]]*?,[^\[\]]*?,[^\[\]]*?,[^\[\]]*?\]", campaign_string)
    return len(matches)

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

