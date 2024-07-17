import yfinance as yf
from datetime import datetime
import numpy as np

def get_data(startdate, enddate, save_path):
    data = yf.download('BTC-USD', start=startdate, end=enddate)
    data.to_csv(save_path)
    return data['Close'].values

if __name__ == "__main__":
    get_data(datetime(2020, 1, 1), datetime(2024, 1, 1), "train_data.csv")