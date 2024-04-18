import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from astropy import units as u
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines

from eztcolors import Colors as C

class Spectrum:
    def __init__(self, data: np.ndarray):
        x = data[:,0]
        y = data[:,1]
        self.fitted_function = None

        # Place the peak at the 100th channel
        x_peak = np.argmax(y)
        y = np.roll(y, int(110 - x_peak), axis=0)

        # Shift the spectrum vertically
        low_mean = np.mean(y[:70])
        y -= low_mean

        self.data = np.stack((x, y), axis=1)

    def plot(self):
        if self.fitted_function:
            x_space = np.linspace(self.data[0,0], self.data[-1,0], 1000)
            plt.plot(x_space*u.Jy, self.fitted_function(x_space*u.um), "r-")
        plt.plot(self.data[:,0], self.data[:,1])
        plt.show()

    def fit(self):
        spec1D = Spectrum1D(spectral_axis=self.data[:,0]*u.um, flux=self.data[:,1]*u.Jy)
        gaussian = models.Gaussian1D(
            amplitude=np.max(self.data[:,1]), mean=(np.argmax(self.data[:,1]))
        )
        try:
            self.fitted_function = fit_lines(
                spec1D, gaussian, 
                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=10000
            )
        except:
            self.fitted_function = 1
    
    def get_FWHM(self) -> np.ndarray:
        assert self.fitted_function, f"{C.RED+C.BOLD}Spectrum must be fitted first.{C.END}"
        
        if self.fitted_function == 1:
            return np.array([np.nan, np.nan])
        else:
            cov_matrix = self.fitted_function.meta["fit_info"]["param_cov"]
            uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
            
            stddev = np.array([
                self.fitted_function.stddev.value,
                uncertainty_matrix[2]
            ])
            return list(stddev * 2*np.sqrt(2*np.log(2)))
    
    def get_residue_stddev(self) -> float:
        assert self.fitted_function, f"{C.RED+C.BOLD}Spectrum must be fitted first.{C.END}"
        if self.fitted_function == 1:
            return np.nan
        else:
            return float(np.std((self.data[:,1]*u.Jy - self.fitted_function(self.data[:,0]*u.um)) / u.Jy))
