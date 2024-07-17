import numpy as np


class Dpo(object):
    def __init__(self, data, period=10):
        self.period = period
        self.data = None

        self.values = self.calc_dpo(data)

    def calc_dpo(self, data):
        dpo = [0] * (self.period - 1)

        for idx in range(self.period, len(data) + 1):
            self.data = data[idx - self.period:idx]
            sma = np.average(self.data)
            dpo.append(data[idx - int(self.period/2)] - sma)

        return np.array(dpo)

    def update_dpo(self, value):
        # update data for calculations
        self.data.append(value)
        self.data.pop(0)

        sma = np.average(self.data)

        return self.data[-(int(self.period/2))] - sma
    

if __name__ == "__main__":
    import pandas
    data = pandas.read_csv("data/trend_train_data.csv")
    data = data['Close']

    dpo = Dpo(data, period=4).values
    # Plotting the DPO values
    import matplotlib.pyplot as plt
    plt.plot(dpo, label='DPO')
    plt.title('Detrended Price Oscillator (DPO)')
    plt.xlabel('Time')
    plt.ylabel('DPO Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("dpo.png")