"""
Pairs Trading Dashboard (Streamlit)
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── 1. CONFIG & STYLING ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, dark, non-corporate aesthetic
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 5% 10% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Highlight neon green for positive, electric pink for negative */
    [data-testid="stMetricDelta"] > div:nth-child(1) {
        color: #39FF14 !important; 
    }
    [data-testid="stMetricDelta"] > div:nth-child(2) {
        color: #FF007F !important;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "pairs_trading_data"

# ─── 2. DATA LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60) # Refreshes every 60 seconds
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# Load Data
live_signals = load_json("live_signals.json")
open_positions = load_json("open_positions.json")
backtest_summary = load_csv("backtest_summary.csv")
signals_df = load_csv("signals.csv")

if not signals_df.empty:
    signals_df["Date"] = pd.to_datetime(signals_df["Date"])

# ─── 3. SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Quant Engine")
st.sidebar.markdown("Real-time Nifty 100 Pairs Trading")
st.sidebar.divider()

if open_positions and "positions" in open_positions:
    total_pnl = sum(p["open_pnl"] for p in open_positions["positions"])
    st.sidebar.metric("Live Open P&L", f"₹{total_pnl:,.0f}")
    st.sidebar.caption(f"Last updated: {open_positions.get('generated_at', 'Unknown')[:16]}")
else:
    st.sidebar.metric("Live Open P&L", "₹0")

# ─── 4. MAIN DASHBOARD ──────────────────────────────────────────────────────────
st.title("📈 Active Market Intelligence")

tab1, tab2, tab3 = st.tabs(["🟢 Live Radar & P&L", "🏆 Backtest Leaderboard", "🔬 Deep Dive Explorer"])

# --- TAB 1: LIVE RADAR ---
with tab1:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Current Open Positions")
        if open_positions and open_positions.get("positions"):
            pos_df = pd.DataFrame(open_positions["positions"])
            
            # Format dataframe for display
            display_df = pos_df[["pair", "position", "open_pnl", "open_pnl_pct"]].copy()
            display_df.columns = ["Pair", "Side", "Unrealised P&L (₹)", "Return (%)"]
            
            # Conditional formatting
            st.dataframe(
                display_df.style.map(
                    lambda x: 'color: #39FF14; font-weight: bold;' if x > 0 else 'color: #FF007F; font-weight: bold;',
                    subset=["Unrealised P&L (₹)", "Return (%)"]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No open positions currently active. Waiting for signals...")

    with col2:
        st.subheader("Real-Time Signals")
        if live_signals and live_signals.get("signals"):
            sig_df = pd.DataFrame(live_signals["signals"])
            
            def color_signal(val):
                if "ENTRY" in val or "SHORT" in val: return 'background-color: rgba(57, 255, 20, 0.2);'
                if "STOP" in val: return 'background-color: rgba(255, 0, 127, 0.2);'
                if "EXIT" in val: return 'color: #FFD700;'
                return ''
                
            st.dataframe(
                sig_df[["pair", "live_z", "signal"]].style.map(color_signal, subset=["signal"]),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Run Phase 3 to generate live signals.")

# --- TAB 2: BACKTEST LEADERBOARD ---
with tab2:
    st.subheader("Historical Strategy Performance")
    if not backtest_summary.empty:
        # Format metrics beautifully
        format_dict = {
            'Total Return (%)': '{:.2f}%',
            'Ann. Return (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}%',
            'Win Rate (%)': '{:.1f}%'
        }
        
        st.dataframe(
            backtest_summary.style.format(format_dict)
            .background_gradient(cmap='viridis', subset=['Sharpe Ratio'])
            .background_gradient(cmap='RdYlGn', subset=['Ann. Return (%)']),
            use_container_width=True,
            height=600
        )
    else:
        st.error("Backtest data missing. Run Phase 4.")

# --- TAB 3: PAIR EXPLORER ---
with tab3:
    if not signals_df.empty:
        pairs_available = signals_df["pair"].unique()
        selected_pair = st.selectbox("Select a Pair to Analyze:", pairs_available)
        
        pair_data = signals_df[signals_df["pair"] == selected_pair].copy()
        
        # Build interactive Plotly chart
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Normalised Prices", "Spread", "Z-Score & Signals"),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Prices
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["s1_price"] / pair_data["s1_price"].iloc[0], 
                                 name=pair_data["t1"].iloc[0], line=dict(color="#00E5FF")), row=1, col=1)
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["s2_price"] / pair_data["s2_price"].iloc[0], 
                                 name=pair_data["t2"].iloc[0], line=dict(color="#FF9100")), row=1, col=1)
        
        # 2. Spread
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["spread"], 
                                 name="Spread", line=dict(color="#B388FF")), row=2, col=1)
        
        # 3. Z-Score
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["z_score"], 
                                 name="Z-Score", line=dict(color="#E0E0E0")), row=3, col=1)
        
        # Add Threshold Lines to Z-Score
        fig.add_hline(y=2.0, line_dash="dash", line_color="#39FF14", row=3, col=1, annotation_text="Entry")
        fig.add_hline(y=-2.0, line_dash="dash", line_color="#39FF14", row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="#757575", row=3, col=1)
        fig.add_hline(y=3.5, line_dash="dot", line_color="#FF007F", row=3, col=1, annotation_text="Stop Loss")
        fig.add_hline(y=-3.5, line_dash="dot", line_color="#FF007F", row=3, col=1)

        # Plot Long/Short signals as markers on the Z-score chart
        longs = pair_data[(pair_data["signal"] == 1) & (pair_data["signal"].shift(1) != 1)]
        shorts = pair_data[(pair_data["signal"] == -1) & (pair_data["signal"].shift(1) != -1)]
        
        fig.add_trace(go.Scatter(x=longs["Date"], y=longs["z_score"], mode="markers", 
                                 marker=dict(symbol="triangle-up", size=10, color="#39FF14"), name="Long Spread"), row=3, col=1)
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["z_score"], mode="markers", 
                                 marker=dict(symbol="triangle-down", size=10, color="#FF007F"), name="Short Spread"), row=3, col=1)

        fig.update_layout(
            height=800, 
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Generate signals (Phase 3) to view the explorer.")