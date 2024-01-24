import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    df = df.iloc[1:, :]
    arr = df.to_numpy()
    return arr.astype(float)


def get_fitted_diffraction():
    m_per_pixel = 0.007 * 11 / 2712
    def diffraction_equation(y, a, y_0, I_0):
        L = 1
        lambd = 650 * 10**(-9)
        return I_0 * (np.sin((np.pi*a*(y-y_0))/(L*lambd)) / (np.pi*a*(y-y_0)/(L*lambd)))**2

    vals = get_csv_data("diffraction/single_slit.csv")
    vals = vals[vals[:,1]<245,:]
    vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))
    a, y_0, I_0 = scipy.optimize.curve_fit(diffraction_equation, vals[:,0], vals[:,1], 
                                           p0=[0.04/1000, 0.0894,300])[0]
    plt.plot(vals[:,0], vals[:,1], "ro", markersize=1)
    y_range = np.linspace(0, np.max(vals[:,0]), 10000)
    print(a, y_0, I_0)
    print(vals)
    plt.plot(y_range, diffraction_equation(y_range, a, y_0, I_0), "g-", markersize=1)
    plt.show()


# get_fitted_diffraction()


def get_fitted_interference():
    m_per_pixel = 0.007 * 11 / 2712
    def interference_equation(y, a, y_0, I_0, d):
        L = 1
        lambd = 650 * 10**(-9)
        return I_0 * (np.sin((np.pi*a*(y-y_0))/(L*lambd)) / (np.pi*a*(y-y_0)/(L*lambd)))**2 * (np.cos((np.pi*d*y)/(L*lambd)))**2

    vals = get_csv_data("interference/single_slit.csv")
    vals = vals[vals[:,1]<245,:]
    vals *= np.tile(np.array([m_per_pixel, 1]), (vals.shape[0],1))
    a, y_0, I_0 = scipy.optimize.curve_fit(interference_equation, vals[:,0], vals[:,1], 
                                           p0=[0.04/1000, 0.0894,300])[0]
    plt.plot(vals[:,0], vals[:,1], "ro", markersize=1)
    y_range = np.linspace(0, np.max(vals[:,0]), 10000)
    print(a, y_0, I_0)
    print(vals)
    plt.plot(y_range, interference_equation(y_range, a, y_0, I_0), "g-", markersize=1)
    plt.show()


get_fitted_interference()
