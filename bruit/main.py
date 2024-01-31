import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
from eztcolors import Colors as C
from pprint import pprint


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[:,3:5].astype(float)


def get_intensity_increase(noise_filename: np.ndarray, reference_filename: np.ndarray):
    noise_data = get_csv_data(noise_filename)
    reference_data = get_csv_data(reference_filename)

    high_low_threshold = 1      # Set the intensity between high and low regimes
    high_data = noise_data[reference_data[:,1] > high_low_threshold]
    low_data = noise_data[reference_data[:,1] < high_low_threshold]

    plt.plot(reference_data[:,0], reference_data[:,1], "ro", markersize=0.5)
    plt.plot(high_data[:,0], high_data[:,1], "go", markersize=0.5)
    plt.plot(low_data[:,0], low_data[:,1], "bo", markersize=0.5)
    # plt.show()

    high_val = ufloat(np.nanmean(high_data[:,1]), np.nanstd(high_data[:,1]))
    low_val = ufloat(np.nanmean(low_data[:,1]), np.nanstd(low_data[:,1]))
    return high_val / low_val


# print(
#     (f"{C.NEGATIVE}Intensity increase{C.END}: " +
#     str(get_intensity_increase(noise_filename='bruit/data/F0002CH1.CSV', reference_filename='bruit/data/F0002CH2.CSV'))
#     ).replace("+/-", " ± ")
# )

pprint(u"{ufloat(3,2)}")


def analyze_multiple_csv():
    pass