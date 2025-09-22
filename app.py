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
    CRITICAL FIX: Using a more robust Regex method.
    The previous method `ast.literal_eval` was too strict and failed.
    This new method uses regex to find all occurrences of a valid campaign ID pattern.
    A valid campaign ID is assumed to be a quoted string containing only letters, numbers, and underscores.
    """
    if not isinstance(campaign_string, str):
        return ""
    
    # Regex to find a string inside single quotes that contains only letters, numbers, and underscores.
    # This specifically targets campaign IDs like 'gmv_123' or 'ro_30' and ignores 'gmv:u0s2.0'.
    campaign_names = re.findall(r"\'([a-zA-Z0-9_]+)\'", campaign_string)
    
    # Join multiple campaign names with a newline character for better display
    return "\n".join(campaign_names)

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

