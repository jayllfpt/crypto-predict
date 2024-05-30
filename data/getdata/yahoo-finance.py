import yfinance as yf
import datetime
btc = yf.download('BTC-USD', start=datetime.datetime(2023, 5, 20), end=datetime.datetime.now())
print(type(btc))
print(btc.head())
btc.to_csv("data/train_data.csv")
