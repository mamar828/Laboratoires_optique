"""
    @file:              FFT.py
    @Author:            Félix Desroches

    @Creation Date:     02/2024
    @Last modification: 02/2024

    @Description:       This file contains functions used to analyse the FFT of a michelson interferometer interferogram
"""

import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[1:,:].astype(float)


def get_fft_from_interferometer(
        data_dir: Tuple[str, List[str]],
        ref_dir: Tuple[str, List[str]],
        plot: bool
) -> Dict[str, Tuple[np.ndarray]]:
    data_dir_path, data_files = data_dir
    ref_dir_path, ref_files = ref_dir

    nb_plots = int(len(data_files if len(data_files) != 0 else os.listdir(data_dir_path)))
    plots_x = int(nb_plots**(1/2))
    plots_y = int(nb_plots // plots_x) if nb_plots % plots_x == 0 else int(nb_plots // plots_x + 1)
    fig, axes = plt.subplots(plots_y, plots_x, sharey="all", sharex="all", )
    # axes[0, 0].tick_params(direction="in")
    # axes[0, 1].tick_params(direction="in")
    # axes[1, 0].tick_params(direction="in")
    # axes[1, 1].tick_params(direction="in")
    fig.supxlabel(r"Longueur d'onde [nm]")
    fig.supylabel("Intensité du signal normalisée [-]")


    for location, text in zip([(i, j) for i in range(2) for j in range(2)], ["a)", "b)", "c)", "d)"]):
        row, column = location
        axes[row, column].text(0.01, 0.88, text, fontsize=12, transform=axes[row, column].transAxes)
    return_dict = {}

    i, j = 0, 0
    for filename in data_files if len(data_files) != 0 else os.listdir(data_dir_path):
        if os.path.isfile(os.path.join(data_dir_path, filename)) and filename != ".DS_Store":
            data = get_csv_data(os.path.join(data_dir_path, filename))
            data_size = np.shape(data)[0]
            if os.path.isfile(os.path.join(ref_dir_path, filename)):
                ref_data = get_csv_data(os.path.join(ref_dir_path, filename))
            else:
                ref_data = np.zeros_like(data)
                warnings.warn(f"{filename} has no eponymous file in the reference directory")

            amplitudes = np.abs(fft(data[:, 2]))[:data_size // 2]
            wavelengths = 1 / fftfreq(data_size, d=2 * (data[1, 1] - data[0, 1]))[:data_size // 2] * 1000
            stop, start = 1000, 350
            domain = slice(np.absolute(wavelengths - stop).argmin(), np.absolute(wavelengths - start).argmin())
            return_dict[filename] = (ref_data[:, 1], ref_data[:, 0], amplitudes[domain], wavelengths[domain])
            axes[i // plots_x, i % plots_x].plot(
                wavelengths[domain],
                amplitudes[domain] / max(amplitudes[domain]),
                c="b"
            )
            if i == 3:
                axes[i // plots_x, i % plots_x].plot(
                    wavelengths[domain],
                    amplitudes[domain]/max(amplitudes[domain]),
                    c="b",
                    label="Spectre de l'interféromètre"
                )
            # axes[i // plots_x, i % plots_x].set_title(filename[:-4])
            i += 1
        else:
            warnings.warn(f"{filename} is not a file and is thus ignored")

    for filename in ref_files if len(ref_files) != 0 else os.listdir(ref_dir_path):
        if os.path.isfile(os.path.join(ref_dir_path, filename)) and filename != ".DS_Store":
            ref_data = get_csv_data(os.path.join(ref_dir_path, filename))
            axes[j // plots_x, j % plots_x].plot(
                ref_data[:, 0],
                ref_data[:, 1] / max(ref_data[:, 1]),
                c="r",
                alpha=0.75
            )
            if j == 3:
                axes[j // plots_x, j % plots_x].plot(
                    ref_data[:, 0],
                    ref_data[:, 1]/max(ref_data[:, 1]),
                    c="r",
                    alpha=0.75,
                    label="Spectre de Ocean Optics"
                )
            # axes[j // plots_x, j % plots_x].set_title(filename[:-4])
            j += 1
        else:
            warnings.warn(f"{filename} is not a file and is thus ignored")
    if plot:
        fig.legend(fontsize=9, loc="upper center", markerscale=9)
        plt.savefig("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/figures/ffts.png", dpi=300, bbox_inches="tight")
        plt.show()
    return return_dict


def find_peaks_in_fft(
        reference_amplitudes: np.ndarray,
        reference_wavelengths: np.ndarray,
        amplitudes: np.ndarray,
        wavelengths: np.ndarray
) -> List[float]:
    data = np.absolute(np.stack((reference_amplitudes, reference_wavelengths)).T)
    # data = np.absolute(np.stack((amplitudes, wavelengths)).T)
    slope = np.diff(data[:, 0])
    average_slope = np.average(abs(slope))
    list_of_wavelengths = []
    for i, slope_value in enumerate(slope):
        if slope_value > 2*average_slope or slope_value < -2*average_slope:
            wavelength_slice = data[max(i - 5, 0):min(i + 5, len(data[:, 0])), 0]
            spike = max(wavelength_slice)
            spike_position = np.absolute(data[:, 0]-spike).argmin()
            if data[spike_position, 1] not in list_of_wavelengths:
                list_of_wavelengths.append(data[spike_position, 1])

    return list_of_wavelengths
        # index_gauche = np.argmin(np.abs(tableau[(index_max - 50):(index_max), 0]-amplitude_max/2)) - 50 + index_max
        # incertitude_gauche = abs(tableau[index_gauche,1] - frequence_recalculee)
        # index_droit = np.argmin(np.abs(tableau[(index_max ):(index_max + 50), 0]-amplitude_max/2)) + index_max
        # incertitude_droit = abs(tableau[index_droit,1] - frequence_recalculee)


ffts = get_fft_from_interferometer(
    (
        "/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/fft_for_figure",
        ["w.csv", "na.csv", "hg.csv", "hene.csv"]
    ),
    (
        "/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/oo_for_figure",
        ["w.csv", "na.csv", "hg.csv", "hene.csv"]
    ),
    False)
for file_name, data_tuples in ffts.items():
    print(file_name)
    print(find_peaks_in_fft(data_tuples[0], data_tuples[1], data_tuples[2], data_tuples[3]))
ffts = get_fft_from_interferometer(
    (
        "/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/fft_for_figure",
        ["w.csv", "na.csv", "hg.csv", "hene.csv"]
    ),
    (
        "/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/oo_for_figure",
        ["w.csv", "na.csv", "hg.csv", "hene.csv"]
    ),
    True)
