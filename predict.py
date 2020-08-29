from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('SpotifyMarketHistory.csv')
df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
df.index = df['date']

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['date', 'close'])
for i in range(0, len(df)):
    new_data['date'][i] = data['date'][i]
    new_data['close'][i] = data['close'][i]

new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:500, :]
valid = dataset[500:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid-closing_price), 2)))

train = new_data[:500]
valid = new_data[500:]
valid['Predictions'] = closing_price
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
