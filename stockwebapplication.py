import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import datetime

# Function to fetch data
def fetch_stock_data(ticker):
    data = yf.download(ticker, start="2004-01-01", end=datetime.datetime.now().strftime('%Y-%m-%d'))
    return data

# Function to plot the closing price and 100 days moving average
def plot_hundred_days_moving_average(data):
    plt.figure(figsize=(10, 6))
    
    # Plotting closing price and moving averages
    plt.plot(data['Close'], label='Closing Price')
    plt.plot(data['Close'].rolling(window=100).mean(), label='100 Days MA')
    
    plt.title('Closing Price vs 100 days Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to plot the closing price and 200 days moving average
def plot_two_hundred_days_moving_average(data):
    plt.figure(figsize=(10, 6))
    
    # Plotting closing price and moving averages
    plt.plot(data['Close'], label='Closing Price')
    plt.plot(data['Close'].rolling(window=200).mean(), label='200 Days MA')
    
    plt.title('Closing Price vs 200 days Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to plot the closing price and moving averages
def plot_moving_averages(data):
    plt.figure(figsize=(10, 6))
    
    # Plotting closing price and moving averages
    plt.plot(data['Close'], label='Closing Price')
    plt.plot(data['Close'].rolling(window=100).mean(), label='100 Days MA')
    plt.plot(data['Close'].rolling(window=200).mean(), label='200 Days MA')
    
    plt.title('Closing Price vs Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))  # Predict next closing price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare the data for training the LSTM model
def prepare_data(data):
    # Use closing prices for prediction
    closing_prices = data['Close'].values
    closing_prices = closing_prices.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices)
    
    # Create sequences of 60 days for training
    X_data = []
    y_data = []
    for i in range(60, len(scaled_data)):
        X_data.append(scaled_data[i-60:i, 0])
        y_data.append(scaled_data[i, 0])
    
    X_data, y_data = np.array(X_data), np.array(y_data)
    
    # Reshape X_data to fit LSTM [samples, time steps, features]
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
    
    return X_data, y_data, scaler

# Function to train the model
def train_lstm_model(data):
    X_data, y_data, scaler = prepare_data(data)
    model = create_lstm_model((X_data.shape[1], 1))
    model.fit(X_data, y_data, epochs=10, batch_size=32)
    return model, scaler

# Function to make a prediction for the next day
def predict_next_day(model, scaler, data):
    # Use the last 60 days for prediction
    last_60_days = data['Close'].values[-60:]
    last_60_days = last_60_days.reshape(-1, 1)
    
    scaled_data = scaler.transform(last_60_days)
    X_test = []
    X_test.append(scaled_data)
    X_test = np.array(X_test)
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    return predicted_price[0][0]


# Function to find related stocks in an uptrend
def find_related_stocks():
    # Dummy function: Can be extended with actual logic (e.g., using Yahoo Finance or other APIs)
    return ["AAPL", "MSFT", "GOOGL"]

# Streamlit app logic
def app():
    st.title("Stock Trend Predictor using LSTM")

    ticker = st.text_input("Enter Stock Ticker", "")
    
    if ticker:
        # Fetch stock data
        stock_data = fetch_stock_data(ticker)

        st.subheader(f"Showing Data for {ticker} (Last 5 rows):")
        st.write(stock_data.tail())

        # Plot 100 days moving average
        st.subheader(f"Closing price vs 100 days moving average to check the performance of the stock in recent time(last 3 months.)")
        plot_hundred_days_moving_average(stock_data)
        st.write("If the closing price surpasses 100 days moving average, it means that it has been performing well recently.")

        # Plot 200 days moving average
        st.subheader(f"Closing price vs 200 days moving average to check the performance of the stock in a long time(last 6 months).")
        plot_two_hundred_days_moving_average(stock_data)
        st.write("If the closing price surpasses 200 days moving average, it means that it has been performing well since a long time.")
        
        # Plot Moving Averages
        st.subheader(f"Closing price vs 100 and 200 days moving averages to check if the stock is in uptrend or downtrend.")
        plot_moving_averages(stock_data)
        st.write("If the stock's closing price surpassed the 100 and 200 day moving averages, then it is experiencing an uptrend; downtrend otherwise. ")
        
        # Train the LSTM model
        model, scaler = train_lstm_model(stock_data)
        
        # Predict next day's price
        predicted_price = predict_next_day(model, scaler, stock_data)
        st.subheader(f"Predicted Price for the Next Day: ${predicted_price:.2f}")
        
        # Recommend related stocks
        related_stocks = find_related_stocks()
        st.subheader(f"Related Stocks in Uptrend: {', '.join(related_stocks)}")

        st.write("Thank you for visiting this app. Happy trading!")
    
    else:
        st.write("Sorry, the stock ticker data is not found.")

# Run the Streamlit app
if __name__ == "__main__":
    app()
