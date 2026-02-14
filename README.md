📈 StockSense Pro – AI-Powered Stock Trend Prediction Dashboard

StockSense Pro is an end-to-end Machine Learning project that predicts short-term stock price direction (UP/DOWN) using historical market data and technical indicators. The trained model is deployed as an interactive fintech-style web dashboard using Streamlit for real-time analysis, visualization, and decision support.

⚠️ This project is built for educational and demonstration purposes only. It is not financial advice.

🚀 Features

🔮 Next-Day Stock Trend Prediction (UP / DOWN)

🟢 Buy / 🔴 Sell / 🟡 Hold Trading Signals (confidence-based)

📊 Interactive Candlestick Charts with Moving Averages

📉 Technical Indicators: RSI, MACD, Volatility, Momentum

🎯 Backtesting Visualization (strategy returns)

📊 Model Evaluation Metrics (Accuracy & Balanced Accuracy)

📥 CSV Export of predictions

🤖 Auto-Demo Mode for presentations/judges

💎 Modern FinTech-Style UI (Dark Theme)

🧠 Machine Learning Approach

Problem Type: Supervised Classification

Target: Predict whether tomorrow’s closing price will be higher than today’s (UP/DOWN)

Model: Random Forest Classifier

Feature Engineering:

Moving Averages (MA10, MA20, MA50)

Volatility (10-day, 20-day)

RSI (Relative Strength Index)

MACD

Momentum Features (Returns over 1, 3, 7 days)

Class Imbalance Handling: class_weight="balanced"

Evaluation Metrics:

Accuracy

Balanced Accuracy

Precision, Recall, F1-score

Due to the stochastic nature of financial markets, short-term stock direction prediction is inherently noisy. The model is intended as a decision-support tool, not a guaranteed trading system.