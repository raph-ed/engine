"""Ultra-Polished Fintech & Quantitative Streamlit Trading Terminal.

Features:
- Bento Grid Dark Glassmorphism Design
- Walk-Forward ML Regime Classification (Purged & Embargoed)
- Ledoit-Wolf Markowitz Optimization
- Black-Scholes Systematic Options Hedging & Greeks
"""

import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.translations import I18N
from modules.data_loader import DEFAULT_TICKERS, load_multi_asset_data
from modules.feature_engineering import build_features_and_targets
from modules.ml_regime_model import WalkForwardRegimeDetector
from modules.portfolio_optimizer import optimize_portfolio, compute_dynamic_regime_allocation
from modules.options_hedging import BlackScholesEngine, simulate_options_hedged_portfolio
from modules.backtester import run_strategy_backtest

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quant Regime & Hedging Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INJECT FUTURISTIC FINTECH CSS (BENTO GRID + GLASSMORPHISM + JETBRAINS MONO) ---
CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E6EDF3;
}

/* Background gradient */
.stApp {
    background-color: #0B0F17;
    background-image: 
        radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.04) 0px, transparent 50%);
}

/* Header & Compliance strip */
.compliance-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 8px;
    padding: 8px 16px;
    margin-bottom: 16px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #94A3B8;
}

.live-badge {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.72rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.live-badge::before {
    content: "";
    display: inline-block;
    width: 6px;
    height: 6px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
}

/* Bento Grid Metric Cards */
.bento-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 16px 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.bento-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}
.bento-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    color: #94A3B8;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.bento-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
}
.bento-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 4px;
}

/* Theme accent colors */
.neon-green { color: #10B981; text-shadow: 0 0 12px rgba(16, 185, 129, 0.3); }
.neon-cyan  { color: #06B6D4; text-shadow: 0 0 12px rgba(6, 182, 212, 0.3); }
.neon-red   { color: #EF4444; text-shadow: 0 0 12px rgba(239, 68, 68, 0.3); }
.neon-amber { color: #F59E0B; text-shadow: 0 0 12px rgba(245, 158, 11, 0.3); }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0E131F !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stMultiSelect {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #94A3B8;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #10B981 !important;
    border-bottom: 2px solid #10B981 !important;
    background: rgba(16, 185, 129, 0.05) !important;
}

/* Methodology Callout Boxes */
.methodo-box {
    background: rgba(17, 24, 39, 0.6);
    border-left: 3px solid #10B981;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin-bottom: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.methodo-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    color: #38BDF8;
    margin-bottom: 6px;
}
.methodo-text {
    font-size: 0.88rem;
    line-height: 1.55;
    color: #CBD5E1;
}
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)


# --- CACHED DATA PIPELINE WITH GRACEFUL ERROR HANDLING ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_market_data(tickers, start_date, force_demo=False):
    return load_multi_asset_data(
        tickers=tickers,
        start_date=start_date,
        end_date=datetime.date.today(),
        force_demo=force_demo,
    )


# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.markdown("### `SYSTEM CONTROL //`")
    lang = st.selectbox("🌐 Interface Language", options=["FR", "EN"], index=0)
    t = I18N[lang]

    st.markdown("---")
    st.markdown(f"**`{t['sidebar_title']}`**")

    feed_mode = st.radio(
        t["data_mode"],
        options=[t["live_feed"], t["demo_feed"]],
        index=0,
    )
    force_demo = (feed_mode == t["demo_feed"])

    if st.button(f"⚡ {t['force_refresh']}", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    selected_assets = st.multiselect(
        t["assets_select"],
        options=["SPY", "QQQ", "EZU", "VGK", "GLD", "TLT", "SHY"],
        default=DEFAULT_TICKERS,
    )
    if not selected_assets:
        selected_assets = DEFAULT_TICKERS

    start_date = st.date_input(
        t["start_date"],
        value=datetime.date.today() - datetime.timedelta(days=365 * 3),
        max_value=datetime.date.today() - datetime.timedelta(days=90),
    )

    ml_choice = st.selectbox(
        t["ml_model"],
        options=["Random Forest", "Logistic Regression"],
        index=0,
    )
    model_type = "rf" if ml_choice == "Random Forest" else "lr"

    st.markdown("---")
    st.markdown("**`PORTFOLIO FRICTIONS & CONSTRAINTS`**")
    max_weight = st.slider(t["max_weight"], min_value=0.15, max_value=0.50, value=0.35, step=0.05)
    tc_bps = st.slider(t["tc_bps"], min_value=0, max_value=30, value=10, step=1)
    slip_bps = st.slider(t["slip_bps"], min_value=0, max_value=20, value=5, step=1)

# --- LOAD DATA ---
with st.spinner("Synchronizing market feeds with 400d warm-up buffer..."):
    df_prices, is_fallback = get_cached_market_data(
        tickers=selected_assets,
        start_date=start_date,
        force_demo=force_demo,
    )

# Feature Engineering & Target Labelling
features_clean, dataset_train, _ = build_features_and_targets(
    df_prices=df_prices,
    target_horizon=20,
    bull_thresh=0.015,
    bear_thresh=-0.02,
    embargo_period=5,
)

# Run Walk-Forward Model
detector = WalkForwardRegimeDetector(model_type=model_type, horizon=20, embargo=5)
regime_preds, regime_probas, feat_importances = detector.run_walk_forward(
    dataset_train=dataset_train,
    all_features=features_clean,
    train_window=252,
    test_step=21,
)

# Filter display range strictly from requested start_date to today
display_mask = df_prices.index >= pd.Timestamp(start_date)
df_display = df_prices.loc[display_mask]
regimes_display = regime_preds.loc[df_display.index]
probas_display = regime_probas.loc[df_display.index]

# Current Live Values (Latest Day)
current_regime = int(regimes_display.iloc[-1])
p_bull = float(probas_display.iloc[-1][0])
p_neutral = float(probas_display.iloc[-1][1])
p_bear = float(probas_display.iloc[-1][2])

latest_vix = float(df_display["^VIX"].iloc[-1]) if "^VIX" in df_display.columns else 18.5
latest_spy = float(df_display["SPY"].iloc[-1])

# Current Regime Badge Styling
if current_regime == 0:
    regime_name = t["regime_0"]
    regime_color_class = "neon-green"
    regime_badge_bg = "rgba(16, 185, 129, 0.15)"
elif current_regime == 1:
    regime_name = t["regime_1"]
    regime_color_class = "neon-cyan"
    regime_badge_bg = "rgba(6, 182, 212, 0.15)"
else:
    regime_name = t["regime_2"]
    regime_color_class = "neon-red"
    regime_badge_bg = "rgba(239, 68, 68, 0.15)"

# --- HEADER & COMPLIANCE BAR ---
utc_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(
    f"""
    <div class="compliance-banner">
        <span>🛡️ {t['compliance_badge']}</span>
        <span class="live-badge">{t['live_sync']} • {utc_now}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title(t["app_title"])
st.caption(f"{t['app_subtitle']} | Engine Status: {'Simulation Fallback' if is_fallback else 'Direct Yahoo Finance API'}")

# --- BENTO GRID KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="bento-card" style="border-left: 3px solid { '#10B981' if current_regime==0 else ('#06B6D4' if current_regime==1 else '#EF4444') };">
            <div>
                <div class="bento-label">{t['kpi_regime']}</div>
                <div class="bento-val {regime_color_class}">{regime_name}</div>
            </div>
            <div class="bento-sub">{t['regime_0_desc'] if current_regime==0 else (t['regime_1_desc'] if current_regime==1 else t['regime_2_desc'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="bento-card">
            <div>
                <div class="bento-label">{t['kpi_bull_prob']}</div>
                <div class="bento-val neon-green">{p_bull * 100:.1f}%</div>
            </div>
            <div class="bento-sub">Confidence Score: { 'High' if p_bull > 0.6 else 'Moderate' }</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="bento-card">
            <div>
                <div class="bento-label">{t['kpi_selloff_prob']}</div>
                <div class="bento-val {'neon-red' if p_bear > 0.3 else 'neon-amber'}">{p_bear * 100:.1f}%</div>
            </div>
            <div class="bento-sub">Tail-Risk Alert: { 'CRITICAL' if p_bear > 0.5 else ('ELEVATED' if p_bear > 0.25 else 'NOMINAL') }</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="bento-card">
            <div>
                <div class="bento-label">{t['kpi_vix']}</div>
                <div class="bento-val {'neon-red' if latest_vix > 25 else ('neon-amber' if latest_vix > 18 else 'neon-cyan')}">{latest_vix:.2f}</div>
            </div>
            <div class="bento-sub">SPY Ref: ${latest_spy:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# Helper function to polish plotly dark layouts
def apply_terminal_layout(fig, height=450):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        margin=dict(l=20, r=20, t=35, b=20),
        font=dict(family="JetBrains Mono, Inter", size=11, color="#94A3B8"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#374151",
            font_size=12,
            font_family="JetBrains Mono",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


# --- 5 STRUCTURED TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_overview"],
    t["tab_markowitz"],
    t["tab_backtest"],
    t["tab_options"],
    t["tab_methodology"],
])

# ==============================================================================
# TAB 1: OVERVIEW & REGIMES
# ==============================================================================
with tab1:
    st.subheader(t["chart_price_regimes"])

    # Price chart overlaid with regime colored markers
    spy_sub = df_display["SPY"]
    fig_regimes = go.Figure()

    fig_regimes.add_trace(
        go.Scatter(
            x=spy_sub.index,
            y=spy_sub.values,
            mode="lines",
            line=dict(color="#64748B", width=1.5),
            name="SPY Price",
        )
    )

    color_map = {0: "#10B981", 1: "#06B6D4", 2: "#EF4444"}
    name_map = {0: "Bullish (R0)", 1: "Neutral (R1)", 2: "Sell-off (R2)"}

    for reg_id in [0, 1, 2]:
        mask = regimes_display == reg_id
        if mask.any():
            fig_regimes.add_trace(
                go.Scatter(
                    x=spy_sub.index[mask],
                    y=spy_sub.values[mask],
                    mode="markers",
                    marker=dict(size=4, color=color_map[reg_id]),
                    name=name_map[reg_id],
                )
            )

    apply_terminal_layout(fig_regimes, height=440)
    st.plotly_chart(fig_regimes, use_container_width=True)

    # Feature Importance Plot
    st.subheader(t["chart_feat_importance"])
    feat_df = feat_importances.sort_values(ascending=True)

    fig_feat = go.Figure(
        go.Bar(
            x=feat_df.values,
            y=feat_df.index,
            orientation="h",
            marker=dict(
                color=feat_df.values,
                colorscale=[[0, "#06B6D4"], [1, "#10B981"]],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
        )
    )
    apply_terminal_layout(fig_feat, height=320)
    st.plotly_chart(fig_feat, use_container_width=True)


# ==============================================================================
# TAB 2: MARKOWITZ & ALLOCATIONS
# ==============================================================================
with tab2:
    st.subheader(t["chart_markowitz_front"])

    active_asset_cols = [c for c in selected_assets if c in df_display.columns]
    returns_display = df_display[active_asset_cols].pct_change().dropna()

    static_w = optimize_portfolio(returns_display, method="max_sharpe", max_weight=max_weight)
    dynamic_w = compute_dynamic_regime_allocation(current_regime, static_w)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(f"**{t['static_alloc']}**")
        df_static = pd.DataFrame(list(static_w.items()), columns=["Asset", "Weight"])
        fig_pie1 = go.Figure(
            data=[
                go.Pie(
                    labels=df_static["Asset"],
                    values=df_static["Weight"],
                    hole=0.55,
                    marker=dict(colors=["#10B981", "#06B6D4", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#64748B"]),
                )
            ]
        )
        fig_pie1.update_layout(template="plotly_dark", height=340, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie1, use_container_width=True)

    with c_right:
        st.markdown(f"**{t['dynamic_alloc']}**")
        df_dyn = pd.DataFrame(list(dynamic_w.items()), columns=["Asset", "Weight"])
        fig_pie2 = go.Figure(
            data=[
                go.Pie(
                    labels=df_dyn["Asset"],
                    values=df_dyn["Weight"],
                    hole=0.55,
                    marker=dict(colors=["#10B981", "#06B6D4", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#64748B"]),
                )
            ]
        )
        fig_pie2.update_layout(template="plotly_dark", height=340, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie2, use_container_width=True)

    st.markdown(
        f"""
        <div class="methodo-box">
            <div class="methodo-title">ESTIMATEUR DE LEDOIT-WOLF SHRINKAGE</div>
            <div class="methodo-text">
                Plutôt que la matrice de covariance empirique instable, les coefficients sont compressés vers un estimateur à structure de marché globale. 
                Plafond institutionnel appliqué : <code>{int(max_weight*100)}%</code> maximum par classe d'actif.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 3: INSTITUTIONAL BACKTEST
# ==============================================================================
with tab3:
    st.subheader(t["backtest_wealth"])

    wealth_df, metrics, n_rebalances = run_strategy_backtest(
        df_prices=df_display,
        regime_series=regimes_display,
        base_assets=active_asset_cols,
        transaction_cost_bps=tc_bps,
        slippage_bps=slip_bps,
        max_weight=max_weight,
    )

    fig_wealth = go.Figure()
    fig_wealth.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df["Strategy"], line=dict(color="#10B981", width=2.5), name=t["strat_hedged"]))
    fig_wealth.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df["Benchmark_6040"], line=dict(color="#06B6D4", width=1.8, dash="dot"), name=t["bench_6040"]))
    fig_wealth.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df["Benchmark_SPY"], line=dict(color="#64748B", width=1.5), name=t["bench_spy"]))

    apply_terminal_layout(fig_wealth, height=440)
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Metrics Matrix Table
    st.subheader(t["perf_summary"])

    table_data = {
        "Metric": [
            t["metric_cagr"],
            t["metric_vol"],
            t["metric_sharpe"],
            t["metric_sortino"],
            t["metric_mdd"],
            t["metric_calmar"],
        ],
        t["strat_hedged"]: [
            f"{metrics['Strategy']['cagr']*100:.2f}%",
            f"{metrics['Strategy']['volatility']*100:.2f}%",
            f"{metrics['Strategy']['sharpe']:.2f}",
            f"{metrics['Strategy']['sortino']:.2f}",
            f"{metrics['Strategy']['max_drawdown']*100:.2f}%",
            f"{metrics['Strategy']['calmar']:.2f}",
        ],
        t["bench_6040"]: [
            f"{metrics['Benchmark_6040']['cagr']*100:.2f}%",
            f"{metrics['Benchmark_6040']['volatility']*100:.2f}%",
            f"{metrics['Benchmark_6040']['sharpe']:.2f}",
            f"{metrics['Benchmark_6040']['sortino']:.2f}",
            f"{metrics['Benchmark_6040']['max_drawdown']*100:.2f}%",
            f"{metrics['Benchmark_6040']['calmar']:.2f}",
        ],
        t["bench_spy"]: [
            f"{metrics['Benchmark_SPY']['cagr']*100:.2f}%",
            f"{metrics['Benchmark_SPY']['volatility']*100:.2f}%",
            f"{metrics['Benchmark_SPY']['sharpe']:.2f}",
            f"{metrics['Benchmark_SPY']['sortino']:.2f}",
            f"{metrics['Benchmark_SPY']['max_drawdown']*100:.2f}%",
            f"{metrics['Benchmark_SPY']['calmar']:.2f}",
        ],
    }

    df_metrics_tbl = pd.DataFrame(table_data)
    st.dataframe(df_metrics_tbl, use_container_width=True, hide_index=True)
    st.caption(f"Frictions déduites : {tc_bps} bps de courtage + {slip_bps} bps slippage par rééquilibrage. Total rotations exécutées : {n_rebalances}.")


# ==============================================================================
# TAB 4: OPTIONS HEDGING & GREEKS
# ==============================================================================
with tab4:
    st.subheader(t["options_title"])
    st.write(t["options_desc"])

    vix_series = df_display["^VIX"] if "^VIX" in df_display.columns else None
    unhedged_w, hedged_w, hedge_pnl = simulate_options_hedged_portfolio(
        spot_prices=df_display["SPY"],
        vix_series=vix_series,
        regime_series=regimes_display,
        hedge_otm_pct=0.05,
        put_duration_days=30,
    )

    fig_hedge = go.Figure()
    fig_hedge.add_trace(go.Scatter(x=unhedged_w.index, y=unhedged_w.values, line=dict(color="#EF4444", width=1.8), name=t["unhedged_spot"]))
    fig_hedge.add_trace(go.Scatter(x=hedged_w.index, y=hedged_w.values, line=dict(color="#10B981", width=2.2), name=t["hedged_spot"]))

    apply_terminal_layout(fig_hedge, height=400)
    st.plotly_chart(fig_hedge, use_container_width=True)

    # Black-Scholes Greeks Instantaneous Display
    st.subheader(t["greeks_title"])
    spot_now = latest_spy
    strike_now = spot_now * 0.95
    t_now = 30 / 365.0
    r_now = 0.03
    sigma_now = max(latest_vix / 100.0, 0.12)

    greeks = BlackScholesEngine.compute_greeks(
        S=spot_now, K=strike_now, T=t_now, r=r_now, sigma=sigma_now, opt_type="put"
    )
    put_cost = BlackScholesEngine.put_price(
        S=spot_now, K=strike_now, T=t_now, r=r_now, sigma=sigma_now
    )

    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        st.markdown(
            f"""
            <div class="bento-card">
                <div>
                    <div class="bento-label">DELTA (Δ)</div>
                    <div class="bento-val neon-cyan">{greeks['delta']:.3f}</div>
                </div>
                <div class="bento-sub">{t['delta_desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with gcol2:
        st.markdown(
            f"""
            <div class="bento-card">
                <div>
                    <div class="bento-label">GAMMA (Γ)</div>
                    <div class="bento-val neon-green">{greeks['gamma']:.4f}</div>
                </div>
                <div class="bento-sub">{t['gamma_desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with gcol3:
        st.markdown(
            f"""
            <div class="bento-card">
                <div>
                    <div class="bento-label">THETA (Θ) / JOUR</div>
                    <div class="bento-val neon-red">${greeks['theta_daily']:.3f}</div>
                </div>
                <div class="bento-sub">{t['theta_desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with gcol4:
        st.markdown(
            f"""
            <div class="bento-card">
                <div>
                    <div class="bento-label">VEGA (ν) / +1% VOL</div>
                    <div class="bento-val neon-amber">${greeks['vega_1pct']:.3f}</div>
                </div>
                <div class="bento-sub">{t['vega_desc']} | Premium: ${put_cost:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 5: METHODOLOGY & INTERVIEW PLAYBOOK
# ==============================================================================
with tab5:
    st.subheader(t["faq_title"])

    faqs = [
        (t["faq_1_q"], t["faq_1_a"]),
        (t["faq_2_q"], t["faq_2_a"]),
        (t["faq_3_q"], t["faq_3_a"]),
        (t["faq_4_q"], t["faq_4_a"]),
    ]

    for q, a in faqs:
        st.markdown(
            f"""
            <div class="methodo-box">
                <div class="methodo-title">{q}</div>
                <div class="methodo-text">{a}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
