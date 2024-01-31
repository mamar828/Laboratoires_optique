from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# guess_dict = {
#     "dif": [910, 1250, 1600, 2000, 2350, 2700, 3000, 3650, 4000, 4350, 4700, 4950, 5250, 5550, 5800, 6100],
#     "int": [470, 1180, 1900, 2600, 2700, 2800, 2900, 3000, 3075, 3200, 3300, 3375, 3475, 3550, 3660, 3750, 3850, 3950, 4000, 4100, 4200, 4300, 4375, 4475, 4550, 4652, 4750, 4800, 4900, 5000, 5150, 5700, 6300, 6700],
#     "cer": [2850, 3050, 3200, 3600, 3790, 3900],
#     "cer_3": [890, 980, 1080, 1180, 1300, 1500, 1630, 1750, 1815, 1950, 2050],
#     "squ": []
# }
M_PER_PIXEL = 2.8e-5


def terminate_figure(
        fig: Optional[plt.Figure] = None,
        show: bool = True,
        path_to_save: Optional[str] = None,
        **kwargs
) -> None:
    """
    Terminates current figure.

    Parameters
    ----------

    fig : plt.Figure
        Current figure. If no figure is given, will close the opened figure.
    show : bool
        Whether to show figure. Defaults to True
    path_to_save : Optional[str]
        Path to save the figure.
    """
    if fig is not None:
        fig.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
    if show:
        plt.show()

    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def find_valleys_from_guess():
    fig, axes = plt.subplots(2, 2, sharex="row", sharey="all")
    fig.supxlabel("Distance relative à une origine arbitraire [m]")
    fig.supylabel("Intensité lumineuse du pixel [-]")
    for experience_type, position in [
        ("dif", (0, 0)),
        ("int", (0, 1)),
        # ("cer", (1, 0)),
        ("squ", (1, 1)),
        ("cer_3", (1, 0))
    ]:
        file_name = f'/Users/felixdesroches/Downloads/{experience_type}.csv'
        df = pd.read_csv(file_name, sep=',', header=None)[1:].astype(float)
        array = df.to_numpy()
        if experience_type == "cer_3":
            array = array[:3500, :]
        axes[position].plot(array[:, 0]*M_PER_PIXEL, array[:, 1], linewidth=1)
        # plt.plot(array[:, 0], array[:, 1])
        # critical_points = []
        # for guess in guess_dict[experience_type]:
        #     critical_points += [guess -50 + np.argmin(array[guess-50:guess+50, 1])]
        # print(critical_points)
        # for i in critical_points:
        #     plt.plot([i, i], [-10, 10], c="r")
        #     plt.plot([i, i], [-10, 260], c="r")

    terminate_figure(fig, False, "/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/diffraction/figures/subplot_figure.png")


find_valleys_from_guess()
version = "current"


if version == "deprecated":
    guess_dict = {
        "dif": [910, 1250, 1600, 2000, 2350, 2700, 3000, 3650, 4000, 4350, 4700, 4950, 5250, 5550, 5800, 6100],
        "int": [470, 1180, 1900, 2600, 2700, 2800, 2900, 3000, 3075, 3200, 3300, 3375, 3475, 3550, 3660, 3750, 3850, 3950, 4000, 4100, 4200, 4300, 4375, 4475, 4550, 4652, 4750, 4800, 4900, 5000, 5150, 5700, 6300, 6700],
        "cer": [2850, 3050, 3200, 3600, 3790, 3900],
        "cer_3": [890, 980, 1080, 1180, 1300, 1500, 1630, 1750, 1815, 1950, 2050],
        "squ": []
    }


    def find_valleys():
        for experience_type, position in [
            ("dif", (0, 0)),
            ("int", (0, 1)),
            # ("cer", (1, 0)),
            ("squ", (1, 1)),
            ("cer_3", (1, 0))
        ]:
            file_name = f'/Users/felixdesroches/Downloads/{experience_type}.csv'
            df = pd.read_csv(file_name, sep=',', header=None)[1:].astype(float)
            array = df.to_numpy()
            # slope = np.diff(df[:, 1])
            # axe, arr = plt.subplots(2, 2)
            # df_avg = [(df[5*i, 1]+df[5*i+1, 1]+df[5*i+2, 1]+df[5*i+3, 1]+df[5*i+4, 1])/5 for i in range(len(df[:, 0])//5)]
            # slope_of_avg = np.diff(df_avg)
            plt.plot(array[:, 0], array[:, 1])
            # arr[1, 0].plot(df[:len(slope[:]), 0], slope[:])
            # arr[0, 1].plot(df[:len(df_avg[:]), 0], df_avg)
            # arr[1, 1].plot(df[:len(slope_of_avg[:]), 0], slope_of_avg[:])
            critical_points = []
            for guess in guess_dict[experience_type]:
                # critical_points += [df.loc(min(array[guess-75:guess+75, 1]))]
                critical_points += [guess -50 + np.argmin(array[guess-50:guess+50, 1])]
            last_i = -1000
            # for i in range(len(slope)-1):

            #     if np.average(slope[i-10:i]) < -1 and np.average(slope[i:i+10]) > 1 and i-last_i > 50:
            #         critical_points += [i]
            #         last_i = i
            # for i in range(25, len(slope)-50):
            #     if np.abs(np.average(slope[i-25:i+25])) < 0.2 and np.average(slope[i-25:i]) < -0.5 and np.average(slope[i:i+25]) > 0.5 and i-last_i > 25:
            #         critical_points += [i]
            #         last_i = i
            # for i in range(5, len(slope_of_avg)-5):
            #     if np.abs(np.average(slope_of_avg[i-5:i+5])) < 0.2 and np.average(slope_of_avg[i-5:i]) < -0.5 and np.average(slope_of_avg[i:i+5]) > 0.5 and i-last_i > 5:
            #         critical_points += [i]
            #         last_i = i
            # slope = slope_of_avg
            # for i in range(5, len(slope)-10):
            #     if np.average(slope[i-5:i]) < -1 and np.average(slope[i:i+5]) > 1 and i-last_i > 5:
            #         critical_points += [i]
            #         last_i = i
            print(critical_points)
            for i in critical_points:
                plt.plot([i, i], [-10, 10], c="r")
                plt.plot([i, i], [-10, 260], c="r")
            plt.show()


    find_valleys()