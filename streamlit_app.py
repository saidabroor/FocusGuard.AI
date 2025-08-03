import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Focus Detection Dashboard")

# Read the CSV log
log_path = "logs/focus_log.csv"
if not os.path.exists(log_path):
    st.warning("No focus log found. Please run video analysis first.")
else:
    df = pd.read_csv(log_path)

    # Convert time to seconds
    def time_to_seconds(t):
        h, m, s = map(int, t.split(":"))
        return h * 3600 + m * 60 + s

    df["start_sec"] = df["Start Time"].apply(time_to_seconds)

    # Map status to numeric values
    status_map = {
        "Focused": 1,
        "Not Focused": 0,
        "No Eyes Detected": -1,
        "No Face Detected": -2
    }
    df["status_value"] = df["Status"].map(status_map)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["start_sec"], df["status_value"], marker='o', linestyle='-')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Focus Status")
    ax.set_title("Focus Status Over Time")
    ax.set_yticks(list(status_map.values()))
    ax.set_yticklabels(list(status_map.keys()))
    st.pyplot(fig)

    # Stats
    st.markdown("### Summary Stats")
    st.write(df["Status"].value_counts())
