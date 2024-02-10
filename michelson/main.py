import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uncertainties import ufloat


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[1:,:].astype(float)


def plot(filename):
    data = get_csv_data(filename)
    plt.plot(data[:,1], data[:,2], "g-", markersize=2)
    plt.show()


# plot("michelson/data_csv/laser.csv")
# plot("michelson/data_csv/tungsten.csv")
# plot("michelson/data_csv/HG1.csv")
    

def find_DPZ():
    # Uncertainties for the relative dpz and the start of the calibration are chosen to be half of the data steps
    data = get_csv_data("michelson/data_csv/tungsten.csv")
    relative_dpz = ufloat(data[np.argmin(data[:,2]),1], np.diff(data[:2,1])/2)
    
    calib_start = ufloat(1.6555700E+5/9.8, np.diff(data[:2,1])/2)
    return relative_dpz + calib_start


# print(f"DPZ: {find_DPZ()} μm")


def make_graph():
    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(9,6)
    DPZ = find_DPZ().nominal_value

    calibrations = [
        1.6493300E+5 / 9.8,
        1.6555700E+5 / 9.8,
        1.5377300E+5 / 9.8
    ]
    
    raw_arrays = [
        get_csv_data("michelson/data_csv/laser.csv"),
        get_csv_data("michelson/data_csv/tungsten.csv"),
        get_csv_data("michelson/data_csv/HG2.csv")
    ]

    aligned_arrays = [raw_array + calibration - DPZ for raw_array, calibration in zip(raw_arrays, calibrations)]

    # Set subplot labels
    for row, text in zip(range(3), ["a)","b)","c)"]):
        axs[row].text(0.01, 0.83, text, fontsize=15, transform=axs[row].transAxes)

    for row_number, array in zip(range(3), aligned_arrays):
        # axs[row_number].set_xlabel("Différence de parcours [unités arbitraires]")
        # axs[row_number].set_ylabel("Intensité du signal [mV]")
        fig.supxlabel(r"Différence de parcours [$\mu$m]")
        fig.supylabel("Intensité du signal [mV]")

        axs[row_number].plot(array[:,1], array[:,2], "bo", markersize=0.25)

    # plt.savefig("michelson/figures/coherent_distance.png", dpi=300, bbox_inches="tight")
    plt.show()


make_graph()
