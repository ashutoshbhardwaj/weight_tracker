import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
import io

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


# ── Scan PDF parser ──────────────────────────────────────────────────────────
def _parse_scan_pdf(path: str) -> dict:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()

    d = {}

    # Weight from header
    m = re.search(r'\d{3} cm ([\d.]+) kg', text)
    if m: d['weight'] = float(m.group(1))

    # LBM, BFM, SMM, Protein, Mineral, TBW  (1st–6th float/High match)
    float_high = re.findall(r'([\d]+\.[\d]+) / High \[', text)
    for i, key in enumerate(['lean_body_mass', 'body_fat_mass', 'skeletal_muscle',
                              'protein', 'mineral', 'total_body_water']):
        if i < len(float_high):
            d[key] = float(float_high[i])

    # Visceral Fat Level
    m = re.search(r'(\d+) / Over Range', text)
    if m: d['visceral_fat_level'] = int(m.group(1))

    # Subcutaneous Fat & Visceral Fat Mass (decimal bracket %)
    dec_bracket = re.findall(r'([\d.]+) \[ \d+\.\d+% \]', text)
    if len(dec_bracket) >= 1: d['subcutaneous_fat'] = float(dec_bracket[0])
    if len(dec_bracket) >= 2: d['visceral_fat_mass'] = float(dec_bracket[1])

    # Visceral Fat Area
    m = re.search(r'(\d+) / High \[50 - 100\]', text)
    if m: d['visceral_fat_area'] = int(m.group(1))

    # BMR + BWI Score (BWI sits on the same line as BMR in the extracted text)
    m = re.search(r'(\d{4}) kCal ([\d.]+)\n10', text)
    if m:
        d['bmr'] = int(m.group(1))
        d['bwi_score'] = float(m.group(2))

    # TEE (second 4-digit kCal value)
    kcals = re.findall(r'(\d{4}) kCal', text)
    if len(kcals) >= 2: d['tee'] = int(kcals[1])

    # Body Fat %
    m = re.search(r'([\d.]+)% / High \[15 - 20\]', text)
    if m: d['body_fat_pct'] = float(m.group(1))

    # Bio Age (appears as standalone number right after ICF label)
    m = re.search(r'INTRACELLULAR FLUID \(ICF\) KG/LBS\n(\d+)\n', text)
    if m: d['bio_age'] = int(m.group(1))

    # Abdominal Circumference
    m = re.search(r'([\d.]+) cm \(Greater than 102 cm\)', text)
    if m: d['abdominal_circ'] = float(m.group(1))

    # Waist-Hip Ratio
    m = re.search(r'([\d.]+) / High \[0\.75 - 0\.9\]', text)
    if m: d['waist_hip_ratio'] = float(m.group(1))

    return d


@st.cache_data(ttl=3600)
def load_scan_data(cache_buster: int = 0) -> pd.DataFrame:
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive = build("drive", "v3", credentials=creds)
    folder_id = st.secrets["google_drive_folder_id"]

    results = drive.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name)",
        orderBy="name",
    ).execute()

    records = []
    for f in results.get("files", []):
        m = re.search(r'Scan (\d{4}-\d{2}-\d{2})\.pdf', f["name"])
        if not m:
            continue
        content = io.BytesIO(drive.files().get_media(fileId=f["id"]).execute())
        data = _parse_scan_pdf(content)
        data['date'] = pd.Timestamp(m.group(1))
        records.append(data)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


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

# ── Body Composition (Scan Data) ──────────────────────────────────────────────
scans = load_scan_data(st.session_state.get("cache_buster", 0))

if not scans.empty:
    st.header("Body Composition Scans")
    st.caption(
        f"{len(scans)} scans · "
        f"{scans['date'].iloc[0].strftime('%b %d, %Y')} → "
        f"{scans['date'].iloc[-1].strftime('%b %d, %Y')} · "
        "Drop a new 'Scan YYYY-MM-DD.pdf' in the project folder and hit ⟳ Refresh"
    )

    first, latest = scans.iloc[0], scans.iloc[-1]

    # Metric cards — row 1
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Body Fat %",      f"{latest['body_fat_pct']:.1f}%",
              delta=f"{latest['body_fat_pct'] - first['body_fat_pct']:.1f}%",
              delta_color="inverse")
    s2.metric("Body Fat Mass",   f"{latest['body_fat_mass']:.1f} kg",
              delta=f"{latest['body_fat_mass'] - first['body_fat_mass']:.1f} kg",
              delta_color="inverse")
    s3.metric("Lean Body Mass",  f"{latest['lean_body_mass']:.1f} kg",
              delta=f"{latest['lean_body_mass'] - first['lean_body_mass']:.1f} kg")
    s4.metric("Visceral Fat Area", f"{int(latest['visceral_fat_area'])} cm²",
              delta=f"{latest['visceral_fat_area'] - first['visceral_fat_area']:.0f} cm²",
              delta_color="inverse")

    # Metric cards — row 2
    s5, s6, s7, s8 = st.columns(4)
    s5.metric("BWI Score",       f"{latest['bwi_score']:.1f}/10",
              delta=f"{latest['bwi_score'] - first['bwi_score']:.1f}")
    s6.metric("Abdominal Circ.", f"{latest['abdominal_circ']:.1f} cm",
              delta=f"{latest['abdominal_circ'] - first['abdominal_circ']:.1f} cm",
              delta_color="inverse")
    s7.metric("Skeletal Muscle", f"{latest['skeletal_muscle']:.1f} kg",
              delta=f"{latest['skeletal_muscle'] - first['skeletal_muscle']:.1f} kg")
    s8.metric("BMR",             f"{int(latest['bmr'])} kCal",
              delta=f"{latest['bmr'] - first['bmr']:.0f} kCal",
              delta_color="inverse")

    st.divider()

    scans['label'] = scans['date'].dt.strftime('%b %d, %Y')
    dates = scans['label']

    # ── Chart 1: Body Composition Split ──────────────────────────────────────
    st.subheader("Body Composition Split")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Lean Body Mass", x=dates, y=scans['lean_body_mass'],
        marker_color="#4C9BE8",
        text=scans['lean_body_mass'].apply(lambda v: f"{v:.1f} kg"),
        textposition="inside", insidetextanchor="middle",
    ))
    fig_comp.add_trace(go.Bar(
        name="Body Fat Mass", x=dates, y=scans['body_fat_mass'],
        marker_color="#FF7043",
        text=scans['body_fat_mass'].apply(lambda v: f"{v:.1f} kg"),
        textposition="inside", insidetextanchor="middle",
    ))
    fig_comp.add_trace(go.Scatter(
        name="Total Weight", x=dates, y=scans['weight'],
        mode="lines+markers+text",
        text=scans['weight'].apply(lambda v: f"{v:.1f} kg"),
        textposition="top center",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=8, color="white"),
    ))
    fig_comp.update_layout(
        barmode="stack",
        xaxis_title="Scan Date", yaxis_title="Mass (kg)",
        height=420, margin=dict(t=10, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    col_l, col_r = st.columns(2)

    # ── Chart 2: Visceral Fat Trend ───────────────────────────────────────────
    with col_l:
        st.subheader("Visceral Fat Trend")
        fig_visc = go.Figure()
        fig_visc.add_hrect(
            y0=50, y1=100, fillcolor="#4CAF50", opacity=0.12, line_width=0,
            annotation_text="Optimal range (50–100 cm²)", annotation_position="top left",
        )
        fig_visc.add_trace(go.Scatter(
            x=dates, y=scans['visceral_fat_area'],
            mode="lines+markers+text",
            text=scans['visceral_fat_area'].apply(lambda v: f"{int(v)} cm²"),
            textposition="top center",
            line=dict(color="#FF7043", width=2), marker=dict(size=10),
            hovertemplate="%{x}<br>Visceral Fat Area: <b>%{y} cm²</b><extra></extra>",
        ))
        fig_visc.update_layout(
            xaxis_title="Scan Date", yaxis_title="Visceral Fat Area (cm²)",
            showlegend=False, height=380, margin=dict(t=10),
        )
        st.plotly_chart(fig_visc, use_container_width=True)

    # ── Chart 3: Body Fat % Timeline ─────────────────────────────────────────
    with col_r:
        st.subheader("Body Fat %")
        fig_bfpct = go.Figure()
        fig_bfpct.add_hrect(
            y0=11, y1=22.9, fillcolor="#4CAF50", opacity=0.12, line_width=0,
            annotation_text="Normal range (11–22.9%)", annotation_position="top left",
        )
        fig_bfpct.add_trace(go.Scatter(
            x=dates, y=scans['body_fat_pct'],
            mode="lines+markers+text",
            text=scans['body_fat_pct'].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            line=dict(color="#FF7043", width=2), marker=dict(size=10),
            hovertemplate="%{x}<br>Body Fat: <b>%{y:.1f}%</b><extra></extra>",
        ))
        fig_bfpct.update_layout(
            xaxis_title="Scan Date", yaxis_title="Body Fat %",
            showlegend=False, height=380, margin=dict(t=10),
        )
        st.plotly_chart(fig_bfpct, use_container_width=True)

    st.divider()

    col_l2, col_r2 = st.columns(2)

    # ── Chart 4: BWI Score ────────────────────────────────────────────────────
    with col_l2:
        st.subheader("BWI Score")
        bwi_colors = ["#F44336" if v < 6 else "#FF7043" if v < 7
                       else "#FFA726" if v < 8 else "#4CAF50"
                       for v in scans['bwi_score']]
        fig_bwi = go.Figure()
        fig_bwi.add_hrect(y0=0,   y1=5.9,  fillcolor="#F44336", opacity=0.06, line_width=0)
        fig_bwi.add_hrect(y0=5.9, y1=6.9,  fillcolor="#FF7043", opacity=0.06, line_width=0)
        fig_bwi.add_hrect(y0=6.9, y1=7.9,  fillcolor="#FFA726", opacity=0.06, line_width=0)
        fig_bwi.add_hrect(y0=7.9, y1=10,   fillcolor="#4CAF50", opacity=0.06, line_width=0)
        for score, label in [(5.9, "Poor"), (6.9, "Below Avg"), (7.9, "Average")]:
            fig_bwi.add_hline(y=score, line_dash="dot", line_color="gray",
                              annotation_text=label, annotation_position="right")
        fig_bwi.add_trace(go.Bar(
            x=dates, y=scans['bwi_score'],
            marker_color=bwi_colors,
            text=scans['bwi_score'].apply(lambda v: f"{v:.1f}/10"),
            textposition="outside",
            hovertemplate="%{x}<br>BWI Score: <b>%{y:.1f}/10</b><extra></extra>",
        ))
        fig_bwi.update_layout(
            xaxis_title="Scan Date", yaxis_title="BWI Score (/10)",
            yaxis_range=[0, 10], showlegend=False, height=380, margin=dict(t=10),
        )
        st.plotly_chart(fig_bwi, use_container_width=True)

    # ── Chart 5: Abdominal Circumference ─────────────────────────────────────
    with col_r2:
        st.subheader("Abdominal Circumference")
        fig_abd = go.Figure()
        fig_abd.add_hline(
            y=102, line_dash="dash", line_color="#4CAF50",
            annotation_text="Risk threshold (102 cm)", annotation_position="bottom right",
        )
        fig_abd.add_trace(go.Scatter(
            x=dates, y=scans['abdominal_circ'],
            mode="lines+markers+text",
            text=scans['abdominal_circ'].apply(lambda v: f"{v:.1f} cm"),
            textposition="top center",
            line=dict(color="#FF7043", width=2), marker=dict(size=10),
            hovertemplate="%{x}<br>Abdominal Circ: <b>%{y:.1f} cm</b><extra></extra>",
        ))
        fig_abd.update_layout(
            xaxis_title="Scan Date", yaxis_title="Circumference (cm)",
            showlegend=False, height=380, margin=dict(t=10),
        )
        st.plotly_chart(fig_abd, use_container_width=True)

    st.divider()

# ── Raw data expander ────────────────────────────────────────────────────────
with st.expander("View raw data"):
    display = df[["Date", "Weight"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Weight"] = display["Weight"].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
