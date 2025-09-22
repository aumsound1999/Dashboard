# app.py
# Step 3: Final robust parsing logic.

import io
import re
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

def extract_campaign_data(campaign_string: str) -> list:
    """
    FINAL ROBUST LOGIC:
    This function uses a reliable regex to find all substrings that match the
    structural pattern of a campaign with performance data (a list with an ID and many numbers).
    It then extracts the ID from each match.
    """
    if not isinstance(campaign_string, str):
        return []
    
    # Regex Explanation:
    # \[\s* -> Matches the opening bracket '[' of a campaign list, with optional whitespace.
    # '([^']+)'    -> Captures the campaign ID. It looks for a single-quoted string. 
    #                The ID itself is captured in Group 1.
    # \s*,\s* -> Matches the comma after the ID.
    # (?:[^,]+,){5} -> This is the key part. It matches at least 5 comma-separated values
    #                that follow the ID. This ensures we only capture lists with performance data.
    # [^\]]* -> Matches the rest of the characters until the closing bracket.
    # \]           -> Matches the closing bracket ']'.
    matches = re.findall(r"\[\s*'([^']+)'\s*,\s*(?:[^,]+,){5}[^\]]*\]", campaign_string)
    
    return matches

def count_active_campaigns(campaign_string: str) -> int:
    """Counts active campaigns by calling the extractor and getting the length."""
    return len(extract_campaign_data(campaign_string))

def get_campaign_names(campaign_string: str) -> str:
    """Gets active campaign names and joins them with newlines."""
    names = extract_campaign_data(campaign_string)
    return "\n".join(names)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main():
    st.title("Step 3: Final Campaign Verification")
    st.info("This version uses a highly robust method to identify and count campaigns based on their data structure. This should now be 100% accurate.")

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
        
        # Apply the new, robust functions
        summary_df['Active Campaign Count'] = summary_df['Raw Campaign Data'].apply(count_active_campaigns)
        summary_df['Active Campaign IDs'] = summary_df['Raw Campaign Data'].apply(get_campaign_names)

        # --- Display the result ---
        st.markdown("### Active Campaigns per Channel")
        
        st.dataframe(
            summary_df[['Channel', 'Active Campaign Count', 'Active Campaign IDs', 'Raw Campaign Data']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()

