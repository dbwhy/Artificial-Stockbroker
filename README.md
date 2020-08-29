# Artificial-Stockbroker

This project is an attempt to 'hack' the stock market via machine learning; the "artificial stockbroker" is split into two main parts: predicting and trading.

The predicting stage utilizes the TensorFlow library to train a recurrent neural network (RNN) model to predict a stock's *closing price*. The RNN architecture employed is the LSTM (Long Short-Term Memory) algorithm.

The trading stage will be responsible for gathering necessary data and buying/selling shares (via E-Trade API) based on a unique "stockbroker" trading algorithm. The "stockbroker" will consider a stock's recent % change and dip/upward trend threshold values in addition to predictions from the model to determine if action is necessary.

*The predicting stage is currently under development. Once an error (RSME) of less than 10% is achieved, I will begin actualizing the trading stage.*
