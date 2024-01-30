import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from eztcolors import Colors as C
import uncertainties

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
#     "diffraction/data/cer.csv",
#     "position [m]",
#     "intensité [-]",
#     ("Intensité en fonction de la position dans une figure de diffraction produite\navec un laser à 650 nm "+
#      "par une ouverture circulaire d'un diamètre de 0.2 mm"),
#     "diffraction/figures/cer.png"
# )


def find_peak_distance(mins: list):
    y = np.diff(mins) * m_per_pixel
    # Filter aberrant data and compute mean value
    real_y = y[y<np.mean(y)+np.std(y)]
    mean_distance = uncertainties.ufloat(np.mean(real_y), np.std(real_y))
    lambd = uncertainties.ufloat(650,1)*1e-9
    L = uncertainties.ufloat(1,0.01)
    distance = lambd / (mean_distance / L)
    print(f"{C.UNDERLINE}Mean min distance{C.END}: {C.BOLD}{C.YELLOW}{mean_distance:.1u} m{C.END}".replace(
                                                                                                        "+/-"," ± "))
    print(f"{C.UNDERLINE}Measure{C.END}:           {C.BOLD}{C.LIGHT_BLUE}{distance:.1u} m{C.END}".replace("+/-"," ± "))


print(f"\n{C.NEGATIVE}Single slit 0.08mm{C.END}")
find_peak_distance([922, 1238, 1592, 1961, 2319, 2665, 2996, 3631, 3975, 4309, 4653, 4953, 5243, 5575, 5841, 6089])

print(f"\n{C.NEGATIVE}Double slits 0.08mm, distance between slits{C.END}")
find_peak_distance([2593, 2693, 2795, 2891, 2990, 3089, 3185, 3276, 3369, 3466, 3555, 3653, 3745, 
                    3840, 3930, 4019, 4112, 4203, 4289, 4379, 4473, 4553, 4646, 4737, 4827, 4911, 5001, 5108])
print(f"\n{C.NEGATIVE}Double slits 0.08mm, slit width{C.END}")
find_peak_distance([503, 1211, 1897, 2593, 5108, 5737, 6265, 6737])

print(f"\n{C.NEGATIVE}Spherical opening 0.2mm, radius{C.END}")
find_peak_distance([845, 958, 1069, 1159, 1277, 1530, 1627, 1798, 1798, 1932, 2034])
