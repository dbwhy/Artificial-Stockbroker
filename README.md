# Artificial-Stockbroker

This project is an attempt to 'hack' the stock market via machine learning and is split into two parts: predicting and trading.

The predicting stage will utilize the TensorFlow library to train a recurrent neural network (RNN) model to predict a stock's *closing price*. The RNN architecture used will be the LSTM (Long Short-Term Memory) algorithm.

The trading stage will be responsible for gathering necessary data and buying/selling shares (via E-Trade API) based on a unique "artificial stockbroker" trading algorithm. The "stockbroker" will consider a stock's recent % change and dip/upward trend threshold values in addition to predictions from the model to determine if action is necessary.

*The predicting stage is currently under development. Once an error (RSME) of less than 10% is achieved, I will begin actualizing the trading stage.*    
