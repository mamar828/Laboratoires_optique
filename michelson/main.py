import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import *
from numpy.fft import *

def get_csv_data(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, sep=',', header=None)
    arr = df.to_numpy()
    return arr[1:,:].astype(float)


def plot(filename):
    data = get_csv_data(filename)
    plt.plot(data[:,0], data[:,2], "g-", markersize=2)
    plt.show()

def fourierTransformInterferogram(x,y):
	spectrum = fft(y)
	dx = x[1]-x[0] # on obtient dx, on suppose equidistant
	N = len(x)     # on obtient N directement des données
	frequencies = fftfreq(N, dx) # Cette fonction est fournie par numpy
	wavelengths = 1/frequencies  # Les fréquences en µm^-1 sont moins utiles que lambda en µm
	return (wavelengths, frequencies, spectrum)

plot("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/data_csv/tungsten.csv")
plot("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/data_csv/laser.csv")
plot("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/data_csv/HG3.csv")
plot("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/data_csv/HG2.csv")
plot("/Users/felixdesroches/Desktop/ULaval_labs/PHY_2006_optique/Laboratoires_optique/Michelson/data_csv/HG1.csv")

# plt.plot(np.random.random(10**6), np.random(10**6))
# plt.show()

# hdulist = open('michelson/data/HG2')
# print(hdulist.read())

# with open("michelson/data/HG1.txt", "r") as file:
#     print(file.read())

# print(get_csv_data("michelson/data_csv/HG1.csv"))
    
