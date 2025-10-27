import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# 🎯 Streamlit App
# -------------------------------
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Prediction using LSTM")
st.markdown("Enter any stock symbol (e.g., **AAPL**, **TSLA**, **RELIANCE.NS**) to predict its next-day closing price.")

# Input from user
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if st.button("Predict"):
    with st.spinner("Fetching data & making prediction..."):
        # Download historical data
        data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
        close_data = data[['Close']]

        # Preprocess
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        # Prepare last 60 days for prediction
        X_input = scaled_data[-60:]
        X_input = X_input.reshape(1, 60, 1)

        # Load trained model
        model = load_model("lstm_stock_model.h5")

        # Predict next day's price
        y_pred = model.predict(X_input)
        y_pred = scaler.inverse_transform(y_pred)
        predicted_price = y_pred[0][0]

        st.success(f"Predicted Next Day Closing Price for **{ticker}**: ${predicted_price:.2f}")

        # Plot the recent data
        st.subheader("📊 Recent 100 Days Price Trend")
        st.line_chart(close_data[-100:])

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & LSTM")

