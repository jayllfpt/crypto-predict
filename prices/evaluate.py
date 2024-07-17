import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from trends import TrendPredict
from prices import PricePredict
import matplotlib.pyplot as plt
import pandas
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def visualize_predictions(actual, predicted, title="Actual vs Predicted Prices"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("evalPrice.png")

def calculate_metrics(actual, predicted):
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(actual, predicted)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100
    
    # R-squared (R²)
    r2 = r2_score(actual, predicted)
    
    # Mean Absolute Scaled Error (MASE)
    naive_forecast = np.roll(actual, 1)[1:]
    naive_mae = mean_absolute_error(actual[1:], naive_forecast)
    mase = mae / naive_mae
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    smape = 100 * np.mean(2 * np.abs(np.array(predicted) - np.array(actual)) / (np.abs(np.array(actual)) + np.abs(np.array(predicted))))
    
    # Root Mean Squared Logarithmic Error (RMSLE)
    rmsle = np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual))**2))
    
    # Collect metrics in a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'MASE': mase,
        'sMAPE': smape,
        'RMSLE': rmsle
    }
    
    return metrics

if __name__=="__main__":
    timesteps=10
    df = pandas.read_csv("data/test_data.csv")
    data = np.array(df['Close'])
    date = df['Date']

    trend_predictor = TrendPredict(timesteps=timesteps,
                             model_path="models/trend_model.h5",
                             scaler_path="models/trend_scaler.dump")

    price_predictor = PricePredict(timesteps=timesteps,
                             model_path="models/price_model.h5",
                             scaler_path="models/price_scaler.dump")
    
    processed_data, trends = trend_predictor.extract_data(data)

    groundtruth = data[30 + timesteps:-1]
    predicts = price_predictor(data[30:-1], trends)
    predicts = [x[0] for x in predicts]

    visualize_predictions(groundtruth, predicts)
    print(len(groundtruth), len(predicts))
    data = [
        ['Date', 'Groundtruth', 'Predict']
    ]
    for i in range(len(groundtruth)):
        data.append([date[40+i], groundtruth[i], predicts[i]])

    with open("price_predictions.csv", 'w', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerows(data)
    metrics = calculate_metrics(groundtruth, predicts)

    data = [['Metric', 'Value']]
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        data.append([metric, value])
    with open("price_evaluation.csv", 'w', encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerows(data)
