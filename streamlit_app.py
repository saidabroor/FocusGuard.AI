import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Focus Detection Timeline Dashboard")

csv_path = "logs/focus_timeline.csv"
if not os.path.exists(csv_path):
    st.warning("No timeline log found. Run video analysis first.")
else:
    df = pd.read_csv(csv_path)

    # Convert time strings to datetime
    df["Start Time"] = pd.to_datetime(df["Start Time"], format="%H:%M:%S")
    df["End Time"] = pd.to_datetime(df["End Time"], format="%H:%M:%S")

    # Create a continuous timeline from Start and End times and status
    times = []
    statuses = []

    for _, row in df.iterrows():
        times.append(row["Start Time"])
        statuses.append(row["Status"])
        times.append(row["End Time"])
        statuses.append(row["Status"])

    # Build DataFrame for step plot
    plot_df = pd.DataFrame({"Time": times, "Status": statuses})

    # Map statuses to numbers for y-axis
    status_map = {
        "Focused": 1,
        "Not Focused": 0,
        "No Eyes Detected": -1,
        "No Face Detected": -2
    }
    plot_df["StatusNum"] = plot_df["Status"].map(status_map)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(plot_df["Time"], plot_df["StatusNum"], where='post')

    # Formatting y axis
    ax.set_yticks(list(status_map.values()))
    ax.set_yticklabels(list(status_map.keys()))

    ax.set_xlabel("Time")
    ax.set_title("Focus Status Over Time")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Summary stats
    st.markdown("### Status Summary")
    st.write(df["Status"].value_counts())

