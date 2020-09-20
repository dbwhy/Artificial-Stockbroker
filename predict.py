import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


class PredictClosingPrice:
    def __init__(self):
        self.spotify_market_history = 'SpotifyMarketHistory.csv'
        self.data = self.load_data()
        self.pct_train = 0.45
        self.T = int(len(self.data) * self.pct_train)
        self.look_back = 60
        self.scale = MinMaxScaler(feature_range=(0, 1))
        self.train, self.validate, self.x_train, self.y_train = self.preprocessing()
        self.loss = 'mean_squared_error'
        self.opt = 'adam'
        self.epochs = 1
        self.batch_size = 1

    # load market data, create dataframe, set index
    def load_data(self):
        df = pd.read_csv(self.spotify_market_history)
        df.sort_index(ascending=True, axis=0, inplace=True)
        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
        df.index = df['date']
        return df[['close']]

    # prepare data for model
    def preprocessing(self):
        # creating train and test sets
        dataset = self.data.values
        train = dataset[0:self.T, :]
        valid = dataset[self.T:, :]

        # normalize data
        ds_scaled = self.scale.fit_transform(dataset)

        # convert data into x_train and y_train
        x, y = [], []
        for i in range(self.look_back, len(train)):
            x.append(ds_scaled[i-self.look_back:i, 0])
            y.append(ds_scaled[i, 0])

        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        return [train, valid, x, y]

    def train_test_model(self):
        # build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))

        model.compile(loss=self.loss, optimizer=self.opt)
        model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

        # format model inputs as X_test
        inputs = self.data[len(self.data) - len(self.validate) - self.look_back:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scale.transform(inputs)

        X_test = []
        for i in range(self.look_back, inputs.shape[0]):
            X_test.append(inputs[i-self.look_back:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # predict closing price
        closing_price = model.predict(X_test)
        closing_price = self.scale.inverse_transform(closing_price)

        # evaluate error
        rms = np.sqrt(np.mean(np.power((self.validate-closing_price), 2)))
        print(f"RMSE: {rms}")

        self.plot_model(closing_price)

    def plot_model(self, y_test):
        train = self.data[:self.T]
        validate = self.data[self.T:]
        validate['predictions'] = y_test
        plt.plot(train['close'])
        plt.plot(validate[['close', 'predictions']])
        plt.show()


if __name__ == '__main__':
    pcp = PredictClosingPrice()
    pcp.train_test_model()
