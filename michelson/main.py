import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[1:,:].astype(float)


def plot(filename):
    data = get_csv_data(filename)
    plt.plot(data[:,0], data[:,2], "g-", markersize=2)
    plt.show()



# plot("michelson/data_csv/tungsten.csv")
# plot("michelson/data_csv/laser.csv")
# plot("michelson/data_csv/HG3.csv")

# plt.plot(np.random.random(10**6), np.random(10**6))
# plt.show()

# hdulist = open('michelson/data/HG2')
# print(hdulist.read())

# with open("michelson/data/HG1.txt", "r") as file:
#     print(file.read())

# print(get_csv_data("michelson/data_csv/HG1.csv"))
    


