import streamlit as st
import pandas as pd
import boto3
import os
from io import StringIO
import time

# -------------------------------
# AWS S3 CONFIG
# -------------------------------
BUCKET = os.environ.get("DISASTEROPS_BUCKET", "my-hackathon-disasterops")
KEY = "DisasterOps/latest.csv"

def fetch_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


# -------------------------------
# Badge Renderer
# -------------------------------
def render_badge(action_level):
    action_level = str(action_level).strip().lower()
    if action_level == "deploy":
        return f"🟥 **DEPLOY**"
    elif action_level == "pre-position":
        return f"🟧 **PRE-POSITION**"
    elif action_level == "monitor":
        return f"🟨 **MONITOR**"
    else:
        return f"⬜ {action_level.upper()}"


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="FEMA DisasterOps Dashboard", layout="wide")

st.title("🌩️ FEMA DisasterOps Live Dashboard")
st.markdown("Streaming data from S3 → FEMA/NOAA/USGS → Unified DisasterOps Feed")

# Auto-refresh every 3 minutes
st_autorefresh = st.experimental_rerun  # compatibility placeholder

st_autorefresh_interval = 3 * 60 * 1000  # 3 minutes in ms
st_autorefresh = st.autorefresh(interval=st_autorefresh_interval, key="refresh")

# Manual refresh button
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

@st.cache_data(ttl=180)  # cache for 3 minutes
def load_data():
    return fetch_csv_from_s3(BUCKET, KEY)

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Could not fetch data from S3: {e}")
    st.stop()

# Show raw table with badges
df_display = df.copy()
df_display["FEMA_Status_Badge"] = df_display["FEMA_Action_Level"].apply(render_badge)

# Order columns (badge first)
cols = ["FEMA_Status_Badge"] + [c for c in df_display.columns if c != "FEMA_Status_Badge"]

st.dataframe(df_display[cols], use_container_width=True)

# Summary counts
st.subheader("📊 Summary (Last Pull)")
summary = df_display["FEMA_Action_Level"].value_counts()
for level, count in summary.items():
    st.markdown(f"- {render_badge(level)} → {count} events")
