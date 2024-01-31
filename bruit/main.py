import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[:,3:5].astype(float)


def analyze_intensities():
    noise_data = get_csv_data("bruit/data/F0002CH1.CSV")
    reference_data = get_csv_data("bruit/data/F0002CH2.CSV")
    high_indices = np.where(reference_data[:,1] > 1)
    # print(high_indices)
    # print(noise_data)
    # print(reference_data)
    plt.plot(reference_data[:,0], reference_data[:,1], "ro", markersize=1)
    print(reference_data[-1,0])
    print(np.array(high_indices) * reference_data[-1,0])
    plt.plot(np.array(high_indices) * reference_data[-1,0], np.zeros(len(high_indices)) + 1, "g<", markersize=0.5)
    plt.show()


analyze_intensities()