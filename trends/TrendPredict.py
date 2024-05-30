import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import json
import joblib
import numpy as np
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.models import load_model # type: ignore

from sklearn.preprocessing import StandardScaler
from technical_analysis.coppock import Coppock
from technical_analysis.dpo import Dpo
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.rsi import StochRsi
from technical_analysis.macd import Macd
from technical_analysis.generate_labels import Genlabels



class TrendPredict():
    def __init__(self, timesteps, model_path, scaler_path):
        
        self.timesteps = timesteps
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = Sequential()
        self.scaler = StandardScaler()
        
        os.makedirs("models", exist_ok=True)


    def __call__(self, data):
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        test_data = self.process_test_data(data)
        predictions = self.model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels
    
    def process_test_data(self, data):
        # Extract features
        macd = Macd(data, 6, 12, 3).values
        stoch_rsi = StochRsi(data, period=14).hist_values
        dpo = Dpo(data, period=4).values
        cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
        inter_slope = PolyInter(data, progress_bar=False).values

        # Truncate bad values
        X = np.array([macd, stoch_rsi, inter_slope, dpo, cop])
        X = np.transpose(X)

        # Scale data
        X = self.scaler.transform(X)

        # Reshape data with timesteps
        reshaped = []
        for i in range(self.timesteps, X.shape[0]):
            reshaped.append(X[i - self.timesteps:i])

        return np.array(reshaped)
    
    def extract_data(self, data):
        # obtain labels
        labels = Genlabels(data, window=65, polyorder=3).labels

        # obtain features
        macd = Macd(data, 6, 12, 3).values
        stoch_rsi = StochRsi(data, period=14).hist_values
        dpo = Dpo(data, period=4).values
        cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
        inter_slope = PolyInter(data, progress_bar=True).values

        # truncate bad values and shift label
        X = np.array([macd[30:-1],
                      stoch_rsi[30:-1],
                      inter_slope[30:-1],
                      dpo[30:-1],
                      cop[30:-1],])

        X = np.transpose(X)
        labels = labels[31:]

        return X, labels

    def adjust_data(self, X, y, split_ratio):
        # count the number of each label
        count_1 = np.count_nonzero(y)
        count_0 = y.shape[0] - count_1
        cut = min(count_0, count_1)

        # save some data for testing
        train_idx = int(cut * split_ratio)

        # shuffle data
        np.random.seed(42)
        shuffle_index = np.random.permutation(X.shape[0])
        X, y = X[shuffle_index], y[shuffle_index]

        # find indexes of each label
        idx_1 = np.argwhere(y == 1).flatten()
        idx_0 = np.argwhere(y == 0).flatten()

        # grab specified cut of each label put them together
        X_train = np.concatenate(
            (X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
        X_test = np.concatenate(
            (X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
        y_train = np.concatenate(
            (y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
        y_test = np.concatenate(
            (y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)

        # shuffle again to mix labels
        np.random.seed(7)
        shuffle_train = np.random.permutation(X_train.shape[0])
        shuffle_test = np.random.permutation(X_test.shape[0])

        X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
        X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

        return X_train, X_test, y_train, y_test

    def shape_data(self, X, y):
        # scale data
        X = self.scaler.fit_transform(X)

        joblib.dump(self.scaler, self.scaler_path)

        # reshape data with timesteps
        reshaped = []
        for i in range(self.timesteps, X.shape[0] + 1):
            reshaped.append(X[i - self.timesteps:i])

        # account for data lost in reshaping
        X = np.array(reshaped)
        y = y[self.timesteps - 1:]

        return X, y

    def build_model(self, input_shape):
        # first layer
        self.model.add(
            LSTM(32, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))

        # second layer
        self.model.add(LSTM(32, return_sequences=False))
        self.model.add(Dropout(0.2))

        # fourth layer and output
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

        # compile layers
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        return self.model

    def train(self, data, split_ratio, epochs, batchsize):
        # load and reshape data
        X, y = self.extract_data(data)
        X, y = self.shape_data(X, y)

        # ensure equal number of labels, shuffle, and split
        self.X_train, self.X_test, y_train, y_test = self.adjust_data(X, y, split_ratio)

        # binary encode for softmax function
        self.y_train, self.y_test = to_categorical(
            y_train, 2), to_categorical(y_test, 2)

        self.build_model(input_shape=(X.shape[1], X.shape[2]))

        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=epochs,
                       batch_size=batchsize,
                       shuffle=True,
                       validation_data=(self.X_test, self.y_test))
        if self.model_path:
            self.model.save(self.model_path)


if __name__ == "__main__":
    predictor = TrendPredict(timesteps=10,
                             model_path="models/trend_model.h5",
                             scaler_path="models/trend_scaler.dump")
    
    # test training phase
    import pandas
    data = pandas.read_csv("data/train_data.csv")
    data = data['Close']
    predictor.train(data, split_ratio=0.8, epochs=20, batchsize=8)

    # test inference phase
    with open('data/test_data.json') as f:
        data = json.load(f)
    data = np.array(data['close'])

    result = predictor(data)
    print(result)