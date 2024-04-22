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

    def plot(self, ax=None):
        if ax:
            if self.fitted_function:
                x_space = np.linspace(self.data[0,0], self.data[-1,0], 1000)
                ax.plot(x_space*u.Jy, self.fitted_function(x_space*u.um), "k-")
            ax.plot(self.data[:,0], self.data[:,1], "o", color="silver", markersize=1.5)
        else:
            if self.fitted_function:
                x_space = np.linspace(self.data[0,0], self.data[-1,0], 1000)
                plt.plot(x_space*u.Jy, self.fitted_function(x_space*u.um), "k-")
            plt.plot(self.data[:,0], self.data[:,1], "o", color="silver", markersize=1.5)
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

    def get_peak_distance(self) -> float:
        assert self.data.shape[0] > 480
        data = np.copy(self.data)[:480,:].reshape(3, 480//3, 2)
        x_peaks = np.argmax(data, axis=1)
        offsets = np.array([0,160,320])
        return np.mean(np.diff(x_peaks[:,1] + offsets))
    
    def get_r_squared(self) -> float:
        assert self.fitted_function, f"{C.RED+C.BOLD}Spectrum must be fitted first.{C.END}"
        # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        residuals = self.data[:,1] - self.fitted_function(self.data[:,0] * u.um) / u.Jy
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.data[:,1] - np.mean(self.data[:,1]))**2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared
