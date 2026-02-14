import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Page Config
# =========================
st.set_page_config(page_title="StockSense Pro", page_icon="📈", layout="wide")

# =========================
# Load Model
# =========================
model = joblib.load("model.pkl")

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main { background-color: #0b0f1a; color: #e5e7eb; }
h1, h2, h3 { color: #38bdf8; }
.buy { color: #22c55e; font-weight: 800; }
.sell { color: #ef4444; font-weight: 800; }
.hold { color: #facc15; font-weight: 800; }
.footer { opacity: 0.6; font-size: 12px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("📈 StockSense Pro – AI Stock Trend Dashboard")
st.caption("Interactive technical charts, ML explainability & strategy backtesting (Educational demo)")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Stock Ticker (e.g., TCS.NS, AAPL)", "TCS.NS")
    period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
    run_btn = st.button("🚀 Run Analysis")

# =========================
# Indicator Functions
# =========================
def add_indicators(df):
    df["Return"] = df["Close"].pct_change()
    df["Return_1"] = df["Return"].shift(1)
    df["Return_3"] = df["Return"].rolling(3).mean()
    df["Return_7"] = df["Return"].rolling(7).mean()

    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    rolling_std = df["Close"].rolling(20).std()
    rolling_mean = df["Close"].rolling(20).mean()
    df["BB_Upper"] = rolling_mean + 2 * rolling_std
    df["BB_Lower"] = rolling_mean - 2 * rolling_std

    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()

# =========================
# Main
# =========================
if run_btn:
    df = yf.download(ticker, period=period)
    if df.empty:
        st.error("No data found for this ticker. Try another symbol.")
        st.stop()

    df = add_indicators(df)

    features = [
        "Open","High","Low","Close","Volume",
        "MA_10","MA_20","MA_50",
        "Volatility_10","Volatility_20",
        "RSI","MACD",
        "Return_1","Return_3","Return_7"
    ]

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    X = df[features]
    latest = X.iloc[-1:]

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]

    if pred == 1 and prob > 0.6:
        signal = "BUY 🟢"
        badge = "buy"
    elif pred == 0 and prob > 0.6:
        signal = "SELL 🔴"
        badge = "sell"
    else:
        signal = "HOLD 🟡"
        badge = "hold"

    st.subheader("🔮 Trading Signal")
    st.markdown(f"<h2 class='{badge}'>{signal} (Confidence: {prob:.2f})</h2>", unsafe_allow_html=True)

    # =========================
    # Candlestick + MA + Bollinger
    # =========================
    st.subheader("📊 Price Action (Candlestick + Trend Bands)")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'
    ))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["MA_10"], name="MA 10"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA 50"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(dash="dot")))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(dash="dot")))
    fig_price.update_layout(template="plotly_dark", height=480)
    st.plotly_chart(fig_price, use_container_width=True)

    # =========================
    # Price + Volume Overlay (Attractive Volume)
    # =========================
    st.subheader("📊 Price vs Trading Activity")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.4))
    fig_vol.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", yaxis="y2"))
    fig_vol.update_layout(
        template="plotly_dark",
        height=360,
        yaxis=dict(title="Volume"),
        yaxis2=dict(title="Price", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # =========================
    # MACD (Bulletproof)
    # =========================
    st.subheader("📉 MACD (Momentum)")
    ema12_plot = df["Close"].ewm(span=12, adjust=False).mean()
    ema26_plot = df["Close"].ewm(span=26, adjust=False).mean()
    macd_plot = (ema12_plot - ema26_plot).astype(float).squeeze()
    macd_signal_plot = macd_plot.ewm(span=9, adjust=False).mean().astype(float).squeeze()
    macd_plot_df = pd.DataFrame({"MACD": macd_plot.values, "MACD Signal": macd_signal_plot.values}, index=df.index)
    st.line_chart(macd_plot_df, height=260)

    # =========================
    # Feature Importance (Attractive)
    # =========================
    st.subheader("🧠 Feature Importance (Model Explainability)")
    imp_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}) \
                .sort_values(by="Importance", ascending=False)
    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        title="Top Features Driving the Model", height=420, template="plotly_dark"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # =========================
    # Strategy vs Buy & Hold
    # =========================
    st.subheader("🎯 Strategy vs Buy & Hold (Cumulative Returns)")
    preds = model.predict(X)
    strategy_returns = (df["Return"] * preds).fillna(0).cumsum()
    buy_hold_returns = df["Return"].fillna(0).cumsum()
    st.line_chart(pd.DataFrame({"ML Strategy": strategy_returns, "Buy & Hold": buy_hold_returns}))

    st.markdown("<div class='footer'>Educational demo • Not financial advice</div>", unsafe_allow_html=True)