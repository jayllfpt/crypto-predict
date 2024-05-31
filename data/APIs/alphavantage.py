import requests
import json

def get_data():
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=R2KPZ9RUJEHYKFCI'
    r = requests.get(url)
    API_output = r.json()

    close = []

    for daily_data in API_output['Time Series (Digital Currency Daily)'].values():
        close.append(float(daily_data['4. close']))

    with open("data/test_data.json", "w", encoding= "utf-8") as f:
        json.dump({"close": close[::-1]}, f, indent=2)

    with open("data/full_test_data.json", "w", encoding= "utf-8") as f:
        json.dump(API_output, f, indent=2)

if __name__ == "__main__":
    get_data()