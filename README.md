# Stock-Trend-Predictor
A stock market based web application that displays essential information about a particular stock the user wants to know about.

Introduction: The stock market is an area wherein the investors look forward to growing their money by investing in shares and hence, becoming shareholders of one or more firms.

Tech-stack used: Python to build the machine learning model, streamlit to deploy the webapp.

Objective of the Project: This project focuses on giving crisp, to-the-point data about a particular stock as per the user's input.

Features: 
i.) A text box for taking the stock ticker (the symbol of how the company is listed in the stock market) of any company.
ii.) Displays previous 5 trading days' information about the stock, including the open,close,high,low,adj close,volume information.
iii.) A plot of closing price versus 100 days moving average is displayed to indicate performance of stock in recent times(intended for short-term investors).
iv.) A plot of closing price versus 200 days moving average is displayed to indicate performance of stock in a long time(intended for long-term investors).
v.)  A plot of closing price versus 100 days moving average versus 200 days moving average is displayed to indicate whether the stock is experiencing an uptrend or downtrend.
vi.) The predicted price of the stock for the next day.
vii.) The model is trained on the previous 20 years of stock data to ensure high level of accuracy that is achievable.
