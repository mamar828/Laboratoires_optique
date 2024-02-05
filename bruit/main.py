import os
import re
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.core
from uncertainties import ufloat
from eztcolors import Colors as C
from matplotlib.lines import Line2D


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[:,3:5].astype(float)


def get_intensity_increase(noise_data: np.ndarray, reference_data: np.ndarray) -> uncertainties.core.Variable:
    # noise_data = get_csv_data(noise_filename)
    # reference_data = get_csv_data(reference_filename)

    high_low_threshold = 1      # Set the intensity between high and low regimes
    high_data = noise_data[reference_data[:,1] > high_low_threshold]
    low_data = noise_data[reference_data[:,1] < high_low_threshold]

    plt.plot(reference_data[:,0], reference_data[:,1], "ro", markersize=0.5, label="Signal de référence")
    plt.plot(high_data[:,0], high_data[:,1], "go", markersize=0.5, label="Régime actif")
    plt.plot(low_data[:,0], low_data[:,1], "bo", markersize=0.5, label="Régime passif")
    plt.xlabel("Temps [unités arbitraires]")
    plt.ylabel("Intensité du signal [V]")
    plt.title(("Intensité du signal en fonction du temps du signal de référence et du signal \n" + 
               "étudié permettant la distinction entre le régime actif et le régime passif"))
    plt.legend(loc="upper left", markerscale=10)
    plt.savefig("bruit/figures/cropping_example.png", dpi=300, bbox_inches="tight")
    # plt.show()

    high_val = ufloat(np.nanmean(high_data[:,1]), np.nanstd(high_data[:,1]))
    low_val = ufloat(np.nanmean(low_data[:,1]), np.nanstd(low_data[:,1]))
    return high_val / low_val


# print(
#     (f"{C.NEGATIVE}Intensity increase{C.END}: " +
#     str(get_intensity_increase(get_csv_data('bruit/data_deprecated/F0002CH1.CSV'), 
#                                get_csv_data('bruit/data_deprecated/F0002CH2.CSV')))
#     )
# )


def analyze_multiple_csv(data_dir: str) -> Union[uncertainties.core.Variable, np.ndarray]:
    # any_patern = re.compile(r"^F\d{4}CH\d.CSV$")
    # nb_files = len([
    #     name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name)) and any_patern.match(name)
    # ])
    noise_patern = re.compile(r"^F\d{4}CH1.CSV$")
    reference_data = []
    noise_data = []
    for noise_file_name in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, noise_file_name)) and noise_patern.match(noise_file_name):
            current_data_patern = re.compile(noise_file_name[:-5]+"2.CSV")
            for ref_file_name in os.listdir(data_dir):
                if os.path.isfile(os.path.join(data_dir, ref_file_name)) and current_data_patern.match(ref_file_name):
                    reference_data.append(get_csv_data(os.path.join(data_dir, ref_file_name)))
                    noise_data.append(get_csv_data(os.path.join(data_dir, noise_file_name)))
    summed_spectrum = np.array(
        [np.nansum(noise_data, axis=0), np.nansum(reference_data, axis=0)]) / len(reference_data)
    np.save(f"{data_dir}/summed_spectrum.npy", summed_spectrum)
    return get_intensity_increase(*summed_spectrum)


# print(f"{analyze_multiple_csv('bruit/data_128'):.1u}")


def make_graph():
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10,8)
    spectrums = [
        np.stack((get_csv_data("bruit/data/F0004CH1.CSV"), get_csv_data("bruit/data/F0004CH2.CSV"))),
        np.load("bruit/data_128/summed_spectrum.npy")
    ]

    # Set subplot labels
    for row, col, text in zip([0,0,1,1], [0,1,0,1], ["a)","b)","c)","d)"]):
        axs[row, col].text(0.1, 0.92, text, fontsize=15, transform=axs[row,col].transAxes)

    for row_number, spectrum in zip(range(2), spectrums):
        noise_data = spectrum[0,:,:]
        reference_data = spectrum[1,:,:]

        high_low_threshold = 1      # Set the intensity between high and low regimes
        high_data = noise_data[reference_data[:,1] > high_low_threshold]
        low_data = noise_data[reference_data[:,1] < high_low_threshold]

        axs[row_number, 0].set_xlabel("Temps [unités arbitraires]")
        axs[row_number, 0].set_ylabel("Intensité du signal [V]")
        axs[row_number, 1].set_xlabel("Intensité du signal [V]")
        axs[row_number, 1].set_ylabel("Nombre de pixels [-]")
        # fig.supxlabel("Temps [unités arbitraires]")
        # fig.supylabel("Intensité du signal [V]")

        axs[row_number, 0].plot(high_data[:70,0], high_data[:70,1], "g-", markersize=0.25, 
                                label="Régime actif" if row_number else None)
        axs[row_number, 0].plot(high_data[70:,0], high_data[70:,1], "g-", markersize=0.25)
        axs[row_number, 0].plot(low_data[:1000,0], low_data[:1000,1], "b-", markersize=0.25, 
                                label="Régime passif" if row_number else None)
        axs[row_number, 0].plot(low_data[1000:,0], low_data[1000:,1], "b-", markersize=0.25)
        # axs[0].plot(high_data[:70,0], high_data[:70,1], "g-", markersize=0.25, label="Régime actif")
        # axs[0].plot(high_data[72:,0], high_data[72:,1], "g-", markersize=0.25)
        # axs[0].plot(low_data[2:1000,0], low_data[2:1000,1], "b-", markersize=0.25, label="Régime passif")
        # axs[0].plot(low_data[1003:,0], low_data[1003:,1], "b-", markersize=0.25)
        # axs[0].legend(fontsize=9, loc="upper left")

        axs[row_number, 1].hist(high_data[:,1], bins=np.histogram_bin_edges(high_data[:,1], bins="fd"), color="g")
        axs[row_number, 1].hist(low_data[:,1], bins=np.histogram_bin_edges(low_data[:,1], bins="fd"), color="b")
        # axs[row_number,0].text(0,0.85,"allo")

    fig.legend(fontsize=9, loc="lower center")
    plt.savefig("bruit/figures/poisson_law.png", dpi=300, bbox_inches="tight")
    # plt.show()


make_graph()
