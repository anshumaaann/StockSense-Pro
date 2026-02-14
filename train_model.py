import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import joblib

# =========================
# 1. Download stock data
# =========================
ticker = "TCS.NS"   # Change: RELIANCE.NS, INFY.NS, AAPL, TSLA
df = yf.download(ticker, start="2014-01-01", end="2024-01-01")

# =========================
# 2. Feature Engineering
# =========================
df["Return"] = df["Close"].pct_change()

# Lag / Momentum Features
df["Return_1"] = df["Return"].shift(1)
df["Return_3"] = df["Return"].rolling(window=3).mean()
df["Return_7"] = df["Return"].rolling(window=7).mean()

# Moving Averages
df["MA_10"] = df["Close"].rolling(window=10).mean()
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

# Volatility
df["Volatility_10"] = df["Return"].rolling(window=10).std()
df["Volatility_20"] = df["Return"].rolling(window=20).std()

# RSI (14-day)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26

# =========================
# 3. Target Variable (Noise-Reduced)
# =========================
threshold = 0.002  # 0.2% move threshold to reduce random noise
df["Target"] = ((df["Close"].shift(-1) - df["Close"]) / df["Close"] > threshold).astype(int)

# =========================
# 4. Clean Data
# =========================
df = df.dropna()

# =========================
# 5. Features & Labels
# =========================
features = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_10", "MA_20", "MA_50",
    "Volatility_10", "Volatility_20",
    "RSI", "MACD",
    "Return_1", "Return_3", "Return_7"
]

X = df[features]
y = df["Target"]

# =========================
# 6. Train-Test Split (Time-Series Safe)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 7. Train Model (Tuned & Balanced)
# =========================
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =========================
# 8. Evaluate
# =========================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print("\n================= MODEL PERFORMANCE =================")
print("Accuracy:", round(acc, 4))
print("Balanced Accuracy:", round(bal_acc, 4))
print(classification_report(y_test, y_pred))

# =========================
# 9. Save Model
# =========================
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")