from TrendPredict import TrendPredict
import numpy as np
import json
import csv
import pandas

if __name__ == "__main__":
    timesteps=10
    df = pandas.read_csv("data/test_data.csv")
    data = df['Close']
    date = df['Date']
    print(len(data))

    trend_predictor = TrendPredict(timesteps=timesteps,
                             model_path="models/trend_model.h5",
                             scaler_path="models/trend_scaler.dump")

    processed_data, groundtruth = trend_predictor.extract_data(data)
    groundtruth = groundtruth[timesteps-1:]
    predicts = trend_predictor(data)
    predicts = predicts[30:]
    print(groundtruth)
    print(predicts)
    print(len(groundtruth), len(predicts))

    data = [
        ['Date', 'Groundtruth', 'Predict']
    ]
    correct = 0
    for i in range(len(predicts)):
        correct += (groundtruth[i] == predicts[i])
        data.append([date[40+i], groundtruth[i], predicts[i]])

    with open("trend_evaluation.csv", 'w', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerows(data)
    print(correct, len(groundtruth), correct/len(groundtruth))