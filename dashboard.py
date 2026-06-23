"""
Pairs Trading Dashboard (Streamlit)
Run with: python -m streamlit run dashboard.py
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimalist dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 6px;
    }
    [data-testid="stMetricDelta"] > div:nth-child(1) {
        color: #56d364 !important; 
    }
    [data-testid="stMetricDelta"] > div:nth-child(2) {
        color: #f85149 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = "pairs_trading_data"

# ─── 2. DATA LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
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

live_signals = load_json("live_signals.json")
open_positions = load_json("open_positions.json")
backtest_summary = load_csv("backtest_summary.csv")
signals_df = load_csv("signals.csv")

if not signals_df.empty:
    signals_df["Date"] = pd.to_datetime(signals_df["Date"])

# ─── 3. SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.title("Quant Engine")
st.sidebar.markdown("Real-time Nifty 100 Pairs Trading")
st.sidebar.divider()

if open_positions and "positions" in open_positions:
    total_pnl = sum(p["open_pnl"] for p in open_positions["positions"])
    st.sidebar.metric("Live Open P&L", f"₹{total_pnl:,.0f}")
    st.sidebar.caption(f"Last updated: {open_positions.get('generated_at', 'Unknown')[:16]}")
else:
    st.sidebar.metric("Live Open P&L", "₹0")
    st.sidebar.caption("No active positions.")

# ─── 4. MAIN DASHBOARD ──────────────────────────────────────────────────────────
st.title("Market Intelligence")

tab1, tab2, tab3 = st.tabs(["Live Radar", "Historical Backtests", "Pair Explorer"])

# --- TAB 1: LIVE RADAR ---
with tab1:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Current Open Positions")
        if open_positions and open_positions.get("positions"):
            pos_df = pd.DataFrame(open_positions["positions"])
            
            display_df = pos_df[["pair", "position", "open_pnl", "open_pnl_pct"]].copy()
            display_df.columns = ["Pair", "Side", "Unrealised P&L (₹)", "Return (%)"]
            
            st.dataframe(
                display_df.style.map(
                    lambda x: 'color: #56d364;' if x > 0 else 'color: #f85149;',
                    subset=["Unrealised P&L (₹)", "Return (%)"]
                ),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No open positions currently active.")

    with col2:
        st.subheader("Real-Time Signals")
        if live_signals and live_signals.get("signals"):
            sig_df = pd.DataFrame(live_signals["signals"])
            
            # STRIP EMOJIS: Remove all non-ASCII characters from the signal column
            sig_df["signal"] = sig_df["signal"].str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip()
            
            def color_signal(val):
                if "ENTRY" in val or "SHORT" in val: return 'color: #56d364;'
                if "STOP" in val: return 'color: #f85149;'
                if "EXIT" in val: return 'color: #e3b341;'
                return ''
                
            st.dataframe(
                sig_df[["pair", "live_z", "signal"]].style.map(color_signal, subset=["signal"]),
                width="stretch",
                hide_index=True
            )
        else:
            st.info("No live signals found. Execute Phase 3 to generate data.")

# --- TAB 2: BACKTEST LEADERBOARD ---
with tab2:
    st.subheader("Strategy Performance Summary")
    if not backtest_summary.empty:
        format_dict = {
            'Total Return (%)': '{:.2f}%',
            'Ann. Return (%)': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}%',
            'Win Rate (%)': '{:.1f}%'
        }
        
        st.dataframe(
            backtest_summary.style.format(format_dict),
            width="stretch",
            height=600,
            hide_index=True
        )
    else:
        st.info("Backtest data missing. Please execute your Phase 4 script to generate 'backtest_summary.csv'.")

# --- TAB 3: PAIR EXPLORER ---
with tab3:
    if not signals_df.empty:
        pairs_available = signals_df["pair"].unique()
        selected_pair = st.selectbox("Select a Pair to Analyze:", pairs_available)
        
        pair_data = signals_df[signals_df["pair"] == selected_pair].copy()
        
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Normalised Prices", "Spread", "Z-Score & Signals"),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Prices
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["s1_price"] / pair_data["s1_price"].iloc[0], 
                                 name=pair_data["t1"].iloc[0], line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["s2_price"] / pair_data["s2_price"].iloc[0], 
                                 name=pair_data["t2"].iloc[0], line=dict(color="#8b949e", width=1.5)), row=1, col=1)
        
        # Spread
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["spread"], 
                                 name="Spread", line=dict(color="#d2a8ff", width=1.5)), row=2, col=1)
        
        # Z-Score
        fig.add_trace(go.Scatter(x=pair_data["Date"], y=pair_data["z_score"], 
                                 name="Z-Score", line=dict(color="#c9d1d9", width=1)), row=3, col=1)
        
        fig.add_hline(y=2.0, line_dash="dash", line_color="#56d364", line_width=1, row=3, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="#56d364", line_width=1, row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="#30363d", line_width=1, row=3, col=1)
        fig.add_hline(y=3.5, line_dash="dot", line_color="#f85149", line_width=1, row=3, col=1)
        fig.add_hline(y=-3.5, line_dash="dot", line_color="#f85149", line_width=1, row=3, col=1)

        longs = pair_data[(pair_data["signal"] == 1) & (pair_data["signal"].shift(1) != 1)]
        shorts = pair_data[(pair_data["signal"] == -1) & (pair_data["signal"].shift(1) != -1)]
        
        fig.add_trace(go.Scatter(x=longs["Date"], y=longs["z_score"], mode="markers", 
                                 marker=dict(symbol="triangle-up", size=8, color="#56d364"), name="Long Spread"), row=3, col=1)
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["z_score"], mode="markers", 
                                 marker=dict(symbol="triangle-down", size=8, color="#f85149"), name="Short Spread"), row=3, col=1)

        fig.update_layout(
            height=750, 
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig)
    else:
        st.info("Historical signals missing. Execute Phase 3 to generate data for the explorer.")
