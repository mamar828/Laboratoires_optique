import os
import re
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.core
from uncertainties import ufloat
from eztcolors import Colors as C


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[:,3:5].astype(float)


def get_intensity_increase(noise_filename: str, reference_filename: str) -> uncertainties.core.Variable:
    noise_data = get_csv_data(noise_filename)
    reference_data = get_csv_data(reference_filename)

    high_low_threshold = 1      # Set the intensity between high and low regimes
    high_data = noise_data[reference_data[:,1] > high_low_threshold]
    low_data = noise_data[reference_data[:,1] < high_low_threshold]

    plt.plot(reference_data[:,0], reference_data[:,1], "ro", markersize=0.5)
    plt.plot(high_data[:,0], high_data[:,1], "go", markersize=0.5)
    plt.plot(low_data[:,0], low_data[:,1], "bo", markersize=0.5)
    plt.show()

    high_val = ufloat(np.nanmean(high_data[:,1]), np.nanstd(high_data[:,1]))
    low_val = ufloat(np.nanmean(low_data[:,1]), np.nanstd(low_data[:,1]))
    return high_val / low_val


# print(
#     (f"{C.NEGATIVE}Intensity increase{C.END}: " +
#     str(get_intensity_increase(noise_filename='bruit/data/F0002CH1.CSV', reference_filename='bruit/data/F0002CH2.CSV'))
#     )
# )


def analyze_multiple_csv(data_dir: str) -> Union[uncertainties.core.Variable, np.ndarray]:
    # any_patern = re.compile(r"^F\d{4}CH\d.CSV$")
    # nb_files = len([
    #     name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name)) and any_patern.match(name)
    # ])
    noise_patern = re.compile(r"^F\d{4}CH1.CSV$")
    intensity_increase_list = []
    for noise_file_name in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, noise_file_name)) and noise_patern.match(noise_file_name):
            current_data_patern = re.compile(noise_file_name[:-5]+"2.CSV")
            for ref_file_name in os.listdir(data_dir):
                if os.path.isfile(os.path.join(data_dir, ref_file_name)) and current_data_patern.match(ref_file_name):
                    intensity_increase_list += [get_intensity_increase(
                        os.path.join(data_dir, noise_file_name),
                        os.path.join(data_dir, ref_file_name)
                    )]
    print(intensity_increase_list)
    return np.mean(intensity_increase_list)


print(f"{analyze_multiple_csv('./data'):.1u}")



def analyze_multiple_csv():
    pass



def make_graph():
    noise_data = get_csv_data("bruit/data/F0002CH1.CSV")
    reference_data = get_csv_data("bruit/data/F0002CH2.CSV")

    high_low_threshold = 1      # Set the intensity between high and low regimes
    high_data = noise_data[reference_data[:,1] > high_low_threshold]
    low_data = noise_data[reference_data[:,1] < high_low_threshold]

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(10,5)

    fig.supxlabel("Temps [unités arbitraires]")
    fig.supylabel("Intensité du signal [V]")
    axs[0].plot(high_data[:70,0], high_data[:70,1], "g-", markersize=0.25, label="Régime actif")
    axs[0].plot(high_data[70:,0], high_data[70:,1], "g-", markersize=0.25)
    axs[0].plot(low_data[:1000,0], low_data[:1000,1], "b-", markersize=0.25, label="Régime passif")
    axs[0].plot(low_data[1000:,0], low_data[1000:,1], "b-", markersize=0.25)
    axs[0].legend()

    axs[1].hist(low_data[:,1], bins=[0.68,0.690,0.7,0.71,0.715,0.725,0.73,0.74])
    axs[1].hist(high_data[:,1], bins=[0.735,0.74,0.75,0.755,0.765,0.77,0.78])

    # plt.savefig()
    plt.show()


# make_graph()
