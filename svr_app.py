
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.title("Polxium SVR Demo - Apple Stock")

# 1) Load data
data = yf.download("AAPL", start="2021-01-01", end="2025-01-01")
data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data.dropna(inplace=True)

# 2) Build features/target
X = data[["Return", "MA5", "MA10"]]
y = data["Close"].shift(-1).dropna()
X = X.iloc[:-1, :]  # align lengths

# 3) Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) SVR model
model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5) Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

# 6) Plot
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(data.index[-len(y_test):], y_test, label="Actual Price")
ax.plot(data.index[-len(y_test):], y_pred, label="Predicted Price")
ax.legend()
st.pyplot(fig)
