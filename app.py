import streamlit as st
import pandas as pd
import plotly.graph_objects as go

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSMiUMISvci4F_Z_WthPAr7PE2WuW4bROyHo-cvW0uszd_Hjywxca9fOGR8gsrKRrmgYc74ySYJk07q"
    "/pub?gid=0&single=true&output=csv"
)

st.set_page_config(
    page_title="Weight Tracker",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(SHEET_URL, parse_dates=["Date"])

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

    # Clip to today — ignore future-dated entries
    df = df[df.index <= pd.Timestamp.today().normalize()]

    # Interpolate interior gaps; drop leading/trailing NaN (no extrapolation)
    df["Weight"] = df["Weight"].interpolate(method="time")
    df = df.dropna(subset=["Weight"])

    return df.reset_index()


# ── Load ────────────────────────────────────────────────────────────────────
df = load_data()

start_weight = df["Weight"].iloc[0]
current_weight = df["Weight"].iloc[-1]
total_loss = start_weight - current_weight
days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
avg_loss_per_week = total_loss / (days / 7) if days > 0 else 0

# ── Header ──────────────────────────────────────────────────────────────────
st.title("⚖️ Weight Tracker Dashboard")
st.caption(
    f"Journey from {df['Date'].iloc[0].strftime('%b %d, %Y')} "
    f"to {df['Date'].iloc[-1].strftime('%b %d, %Y')} · {days} days"
)

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
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    margin=dict(t=10),
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
weekly["Week"] = weekly["Date"].dt.strftime("w/c %b %d")
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

# ── Raw data expander ────────────────────────────────────────────────────────
with st.expander("View raw data"):
    display = df[["Date", "Weight"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Weight"] = display["Weight"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
