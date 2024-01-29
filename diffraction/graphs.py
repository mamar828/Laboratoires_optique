import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
m_per_pixel = 2.8e-5                # 0.007 * 11 / 2712


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    df = df.iloc[1:, :]
    arr = df.to_numpy()
    return arr.astype(float)


def diffraction_equation(y, a, y_0, I_0, L, lambd):
    return I_0 * (np.sin((np.pi*a*(y-y_0))/(L*lambd)) / (np.pi*a*(y-y_0)/(L*lambd)))**2


def get_fitted_diffraction():

    vals = get_csv_data("diffraction/diffraction_beau_avg.csv")
    # np.savetxt('testi.txt', vals, fmt='%1.3f')
    vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))

    L, lambd = 1, 650e-9
    a, y_0, I_0 = scipy.optimize.curve_fit(lambda y, a, y_0, I_0: diffraction_equation(y, a, y_0, I_0, L, lambd),
                                        vals[:,0], vals[:,1], p0=[0.5/1000, 0.0894, 170])[0]

    plt.plot(vals[:,0], vals[:,1], "ro", markersize=1)
    y_range = np.linspace(0, np.max(vals[:,0]), 10000)
    print(a, y_0, I_0)
    plt.plot(y_range, diffraction_equation(y_range, a, y_0, I_0, L, lambd), "g-", markersize=1)
    plt.show()


# get_fitted_diffraction()


def save_figure(
        filename: str,
        x_label: str=None,
        y_label: str=None,
        title: str=None,
        save_name: str=None
    ):
    vals = get_csv_data(filename)
    vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))

    plt.plot(vals[:,0], vals[:,1], "b-", markersize=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    else:
        plt.show()


# save_figure(
#     "diffraction/data/Diffraction_smol.csv",
#     "position [m]",
#     "intensité [-]",
#     "Intensité mesurée en fonction de la position dans une\nfigure de diffraction produite par un laser à 650 nm",
#     # "diffraction/figures/diffraction_smol.png"
# )


def find_width(filename):
    vals = get_csv_data(filename)
    # vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))
    plt.plot(vals[:,0], vals[:,1], "b-")
    plt.show()





find_width("diffraction/data/Diffraction_smol.csv")


# def get_interference_figure():
#     vals = get_csv_data("diffraction/interference_beau_avg.csv")
#     vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))

#     plt.plot(vals[:,0], vals[:,1], "r-", markersize=1)
#     plt.show()


# get_interference_figure()
