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
    # All values are given in μm
    data = get_csv_data("michelson/data_csv/tungsten.csv")
    unc = (6.4087213 - -3.9889987) / 1.0280000E+3 / 2 * 10          # times 10 to correct the uncertainty
    relative_dpz = ufloat(data[np.argmin(data[:,2]),1], unc)
    calib_start = ufloat(1.6555700E+5/9.8 + 3.9889987, unc)
    return relative_dpz + calib_start


# print(f"DPZ: {find_DPZ()} μm")
# DPZ: 16898.51 ± 0.07 μm


def make_graph():
    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(9,6)
    DPZ = find_DPZ().nominal_value

    abs_calib = np.array([
        1.6493300E+5 / 9.8,
        1.6555700E+5 / 9.8,
        1.5377300E+5 / 9.8
    ])

    zero_point = np.array([
        1.0306601E+1,
        3.9889987,
        6.8855788E+1
    ])

    x_offsets = abs_calib + zero_point - DPZ
    
    # Get arrays and remove the relative calibrations to calibrate in absolute
    raw_arrays = [
        get_csv_data("michelson/data_csv/laser.csv"),
        get_csv_data("michelson/data_csv/tungsten.csv"),
        get_csv_data("michelson/data_csv/HG2.csv")
    ]

    aligned_arrays = []
    for raw_array, x_offset in zip(raw_arrays, x_offsets):
        x_size = raw_array.shape[0]
        aligned_arrays.append(np.tile([0,x_offset,0], (x_size,1)) + raw_array)
    
    # Set subplot labels
    for row, text in zip(range(3), ["a)","b)","c)"]):
        axs[row].text(0.01, 0.83, text, fontsize=15, transform=axs[row].transAxes)

    for row_number, array in zip(range(3), aligned_arrays):
        # axs[row_number].set_xlabel("Différence de parcours [unités arbitraires]")
        # axs[row_number].set_ylabel("Intensité du signal [mV]")
        fig.supxlabel(r"Différence de parcours [$\mu$m]")
        fig.supylabel("Intensité du signal [mV]")

        axs[row_number].plot(array[:,1], array[:,2], "b-", markersize=0.25)

    # plt.savefig("michelson/figures/coherent_distance.png", dpi=300, bbox_inches="tight")
    plt.show()


# make_graph()


def find_tungsten_coherent_distance():
    shape = 200, 5
    threshold = 3
    data = get_csv_data("michelson/data_csv/tungsten.csv")[:-28,:]     # Get only 1000 points
    mean_intensity = np.mean(data[:,2].reshape(*shape), axis=1)
    mean_position = np.mean(data[:,1].reshape(*shape), axis=1)[:-1]

    print(mean_position[np.abs(np.diff(mean_intensity)) > threshold])
    
    unc = 0.3
    print(ufloat(2, unc) - ufloat(-2, unc))

# find_tungsten_coherent_distance()
