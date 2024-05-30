import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../trends')))

import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model  # type: ignore
from technical_analysis.generate_labels import Genlabels
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM  # type: ignore
from sklearn.preprocessing import MinMaxScaler



class PricePredict:
    def __init__(self, timesteps, model_path, scaler_path):
        self.timesteps = timesteps
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = Sequential()
        self.scaler = MinMaxScaler()
        os.makedirs("models", exist_ok=True)

    def __call__(self, data, trends, result_img_path="price-prediction.png"):
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        test_data = self.process_test_data(data, trends)
        predictions = self.model.predict(test_data)
        predictions = self.scaler.inverse_transform(predictions)

        if result_img_path:
            self.visualize(data, predictions, result_img_path)
        return predictions

    def visualize(self, data, predictions, result_img_path):
        print(len(data), len(predictions))
        plt.plot(data[self.timesteps:], color="black", label="actual")
        plt.plot(predictions, color="green", label="predict")
        plt.xlabel("Time")
        plt.ylabel("price")
        plt.legend(loc="upper left")
        plt.savefig(result_img_path)

    def process_test_data(self, data, trends):
        scaled_data = data.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(scaled_data)
        trends = trends.reshape(-1, 1)
        combined_data = np.hstack((scaled_data, trends))
        print(len(combined_data))
        test_data = []
        for x in range(self.timesteps, len(combined_data)):
            test_data.append(combined_data[x-self.timesteps:x])
        test_data = np.array(test_data)
        return test_data

    def build_model(self, input_shape):
        self.model.add(
            LSTM(32, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def train(self, data, epochs, batchsize):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        joblib.dump(self.scaler, self.scaler_path)
        trends = Genlabels(data, window=65, polyorder=3).labels

        self.X_train, self.y_train = [], []
        trends = trends.reshape(-1, 1)
        combined_data = np.hstack((scaled_data, trends))

        for x in range(self.timesteps, len(combined_data)):
            self.X_train.append(combined_data[x-self.timesteps:x])
            self.y_train.append(scaled_data[x, 0])

        self.X_train, self.y_train = np.array(
            self.X_train), np.array(self.y_train)

        self.build_model(input_shape=(
            self.X_train.shape[1], self.X_train.shape[2]))

        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=epochs,
                       batch_size=batchsize)
        if self.model_path:
            self.model.save(self.model_path)


if __name__ == "__main__":
    predictor = PricePredict(timesteps=10,
                             model_path="models/price_model.h5",
                             scaler_path="models/price_scaler.dump")

    # test training phase
    import pandas
    data = pandas.read_csv("data/train_data.csv")
    data = np.array(data['Close'])
    predictor.train(data, epochs=20, batchsize=8)

    # test inference phase
    with open('data/test_data.json') as f:
        data = json.load(f)
    data = np.array(data['close'])
    trends = Genlabels(data, window=65, polyorder=3).labels
    result = predictor(data, trends)
    # print(result)
