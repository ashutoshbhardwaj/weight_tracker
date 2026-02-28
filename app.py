import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSMiUMISvci4F_Z_WthPAr7PE2WuW4bROyHo-cvW0uszd_Hjywxca9fOGR8gsrKRrmgYc74ySYJk07q"
    "/pub?gid=0&single=true&output=csv"
)

import time


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    # Allow authentication via ?pwd=... query parameter
    if st.query_params.get("pwd") == st.secrets["password"]:
        st.session_state["authenticated"] = True
        return True
    st.title("⚖️ AB's Transformation Journey - Weight Tracker")
    pwd = st.text_input("Enter password", type="password")
    if pwd:
        if pwd == st.secrets["password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not check_password():
    st.stop()

st.set_page_config(
    page_title="Weight Tracker",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_data(cache_buster: int = 0) -> pd.DataFrame:
    url = f"{SHEET_URL}&cb={cache_buster}"
    df = pd.read_csv(url, parse_dates=["Date"])

    # Drop rows with no date, coerce weight to numeric
    df = df.dropna(subset=["Date"])
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # Average weights for duplicate dates
    df = df.groupby("Date", as_index=False)["Weight"].mean()

    # Reindex to a continuous daily range so gaps become NaN rows
    df = df.set_index("Date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "Date"

    # Clip to today (AEST = UTC+10) — ignore future-dated entries
    today_aest = (pd.Timestamp.now('UTC').replace(tzinfo=None) + pd.Timedelta(hours=10)).normalize()
    df = df[df.index <= today_aest]

    # Interpolate interior gaps; drop leading/trailing NaN (no extrapolation)
    df["Weight"] = df["Weight"].interpolate(method="time")
    df = df.dropna(subset=["Weight"])

    return df.reset_index()


# ── Load ────────────────────────────────────────────────────────────────────
df = load_data(st.session_state.get("cache_buster", 0))

start_weight = df["Weight"].iloc[0]
current_weight = df["Weight"].iloc[-1]
total_loss = start_weight - current_weight
days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
avg_loss_per_week = total_loss / (days / 7) if days > 0 else 0

# ── Header ──────────────────────────────────────────────────────────────────
st.title("⚖️ AB's Transformation Journey - Weight Timeline")

col_caption, col_refresh = st.columns([6, 1])
with col_caption:
    st.caption(
        f"Journey from {df['Date'].iloc[0].strftime('%b %d, %Y')} "
        f"to {df['Date'].iloc[-1].strftime('%b %d, %Y')} · {days} days"
    )
with col_refresh:
    if st.button("⟳ Refresh", use_container_width=True):
        st.session_state["cache_buster"] = int(time.time())
        st.rerun()

# ── Metric cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Start Weight", f"{start_weight:.1f} kg")
c2.metric("Current Weight", f"{current_weight:.1f} kg",
          delta=f"{current_weight - start_weight:.1f} kg")
c3.metric("Total Lost", f"{total_loss:.1f} kg")
c4.metric("Avg Loss / Week", f"{avg_loss_per_week:.2f} kg")

st.divider()

# ── Daily progress chart ─────────────────────────────────────────────────────
st.subheader("Daily Weight Progress")

df["7d_avg"] = df["Weight"].rolling(7, center=True, min_periods=1).mean()

fig_daily = go.Figure()
fig_daily.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Weight"],
    mode="lines+markers",
    name="Daily Weight",
    line=dict(color="#4C9BE8", width=2),
    marker=dict(size=5),
    hovertemplate="%{x|%b %d}<br><b>%{y:.1f} kg</b><extra></extra>",
))
fig_daily.add_trace(go.Scatter(
    x=df["Date"],
    y=df["7d_avg"],
    mode="lines",
    name="7-day Average",
    line=dict(color="#FF7043", width=2, dash="dash"),
    hovertemplate="%{x|%b %d}<br>7d avg: <b>%{y:.1f} kg</b><extra></extra>",
))
fig_daily.update_layout(
    xaxis_title="Date",
    yaxis_title="Weight (kg)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    height=420,
    margin=dict(t=10, b=60),
)
st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

# ── Weekly loss chart ────────────────────────────────────────────────────────
st.subheader("Weekly Weight Change")

weekly = (
    df.set_index("Date")["Weight"]
    .resample("W-MON", label="left", closed="left")
    .mean()
    .reset_index()
)
weekly["Change"] = weekly["Weight"].diff()   # negative = lost weight, positive = gained
weekly = weekly.dropna(subset=["Change"])
weekly["Week"] = weekly["Date"].dt.strftime("%b %d")
weekly["Color"] = weekly["Change"].apply(
    lambda x: "#F44336" if x > 0 else "#4CAF50"
)

fig_weekly = go.Figure()
fig_weekly.add_trace(go.Bar(
    x=weekly["Week"],
    y=weekly["Change"],
    marker_color=weekly["Color"],
    text=weekly["Change"].apply(lambda x: f"{x:+.2f} kg"),
    textposition="outside",
    hovertemplate="%{x}<br>Change: <b>%{y:+.2f} kg</b><extra></extra>",
))
fig_weekly.add_hline(y=0, line_width=1, line_color="gray")
fig_weekly.update_layout(
    xaxis_title="Week",
    yaxis_title="Weight Change (kg)",
    showlegend=False,
    height=380,
    margin=dict(t=10),
)
st.plotly_chart(fig_weekly, use_container_width=True)

st.divider()

# ── Monthly change chart ──────────────────────────────────────────────────────
st.subheader("Monthly Weight Change")

today_aest = (pd.Timestamp.now('UTC').replace(tzinfo=None) + pd.Timedelta(hours=10)).normalize()
current_month_start = today_aest.replace(day=1)

# First and last recorded weight per calendar month
monthly_first = df.set_index("Date")["Weight"].resample("MS").first()
monthly_last  = df.set_index("Date")["Weight"].resample("MS").last()
monthly = pd.DataFrame({"First": monthly_first, "Last": monthly_last}).reset_index()
monthly["Change"] = monthly["Last"] - monthly["First"]
monthly["Month"]  = monthly["Date"].dt.strftime("%b %Y")

past_months = monthly[monthly["Date"] < current_month_start].copy()
curr_row    = monthly[monthly["Date"] == current_month_start]

# Trend-based projection for remaining days in current month
actual_change    = 0.0
proj_additional  = 0.0
curr_month_label = current_month_start.strftime("%b %Y")

current_data = df[df["Date"] >= current_month_start].copy()
last_day_of_month = (current_month_start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

if not current_data.empty:
    actual_change = float(current_data["Weight"].iloc[-1] - current_data["Weight"].iloc[0])
    last_entry    = current_data["Date"].iloc[-1]
else:
    last_entry = current_month_start - pd.Timedelta(days=1)

# Slope always derived from last 14 days (spans into prior month if needed)
trend_data = df[df["Date"] >= today_aest - pd.Timedelta(days=13)].copy()
if len(trend_data) >= 2:
    x = (trend_data["Date"] - trend_data["Date"].iloc[0]).dt.days.values.astype(float)
    slope = float(np.polyfit(x, trend_data["Weight"].values, 1)[0])  # kg per day
else:
    slope = 0.0

days_remaining  = int((last_day_of_month - last_entry).days)
proj_additional = slope * days_remaining

fig_monthly = go.Figure()

# Past months — solid bars
fig_monthly.add_trace(go.Bar(
    x=past_months["Month"],
    y=past_months["Change"],
    marker_color=past_months["Change"].apply(lambda v: "#F44336" if v > 0 else "#4CAF50"),
    name="Monthly Change",
    text=past_months["Change"].apply(lambda v: f"{v:+.2f} kg"),
    textposition="outside",
    hovertemplate="%{x}<br>Change: <b>%{y:+.2f} kg</b><extra></extra>",
))

# Current month — actual so far (solid)
actual_color = "#F44336" if actual_change > 0 else "#4CAF50"
fig_monthly.add_trace(go.Bar(
    x=[curr_month_label],
    y=[actual_change],
    marker_color=actual_color,
    name="Current (actual)",
    text=[f"{actual_change:+.2f} kg"],
    textposition="outside",
    hovertemplate="%{x}<br>Actual so far: <b>%{y:+.2f} kg</b><extra></extra>",
))

# Current month — projected remaining (semi-transparent, stacked)
if days_remaining > 0:
    proj_color = "rgba(76,175,80,0.35)" if proj_additional <= 0 else "rgba(244,67,54,0.35)"
    fig_monthly.add_trace(go.Bar(
        x=[curr_month_label],
        y=[proj_additional],
        marker_color=proj_color,
        marker_line=dict(color="#888888", width=1),
        name="Projected",
        hovertemplate=(
            f"{curr_month_label}<br>"
            f"Projected additional: <b>{proj_additional:+.2f} kg</b>"
            f"<br>Expected month total: <b>{actual_change + proj_additional:+.2f} kg</b>"
            "<extra></extra>"
        ),
    ))

fig_monthly.add_hline(y=0, line_width=1, line_color="gray")
fig_monthly.update_layout(
    barmode="relative",
    xaxis_title="Month",
    yaxis_title="Weight Change (kg)",
    height=380,
    margin=dict(t=10, b=60),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
)
st.plotly_chart(fig_monthly, use_container_width=True)

st.divider()

# ── Raw data expander ────────────────────────────────────────────────────────
with st.expander("View raw data"):
    display = df[["Date", "Weight"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Weight"] = display["Weight"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
