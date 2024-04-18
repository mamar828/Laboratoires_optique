from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from eztcolors import Colors as C


class Map:
    def __init__(self, data):
        self.data = data

    @classmethod
    def load_from_file(cls, filename):
        file_data = fits.open(filename)
        data = np.array([array.data for array in file_data])
        return cls(data)
    
    def save(self, filename):
        hdu_list = fits.HDUList([
            fits.PrimaryHDU(self.data[0,:,:]),
            *[fits.ImageHDU(self.data[i,:,:]) for i in range(1, self.data.shape[0])]
        ])
        try:
            hdu_list.writeto(filename, overwrite=False)
        except OSError:
            while True:
                decision = input(f"{C.RED}{filename} already exists, do you wish to overwrite it? [y/n]{C.END}")
                if decision.lower() == "y":
                    hdu_list.writeto(filename, overwrite=True)
                    break
                elif decision.lower() == "n":
                    break
    
    def plot(self, index=0, save_filename=None):
        plt.imshow(self.data[index,:,:])
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename, dpi=600)
        else:
            plt.show()
