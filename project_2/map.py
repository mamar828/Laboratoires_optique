from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from eztcolors import Colors as C


class Array:
    def __init__(self, data):
        self.data = data

    def plot(self, save_filename=None):
        plt.colorbar(plt.imshow(self.data[:,:]))
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename, dpi=600)
        else:
            plt.show()


class ValueMap: ...

class Map:
    def __init__(self, data: np.ndarray, type: str, name: str, units: str):
        self.data = data
        self.type = type
        self.name = name
        self.units = units

    def __type__(self):
        return self.type

    def get_ImageHDU(self):
        return fits.ImageHDU(self.data, fits.Header({'type': self.type, 'name': self.name, 'units': self.units}))
    
    def plot(self, save_filename=None):
        plt.colorbar(plt.imshow(self.data[:,:]))
        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename, dpi=600)
        else:
            plt.show()


class Maps(dict):
    def __init__(self, maps: list[Map]):
        # self.maps = {
        #     map_.name : map_.name.get{map_.type: map_} for map_ in maps
        # }
        print(self.maps)

    # def __getitem__(self, name: str):
    #     assert (name in self.maps.keys(),
    #             f"{C.RED+C.BOLD}{name} does not exist. Existing names are {list(self.maps.keys())}")
    #     return self.maps[name]
    
    def get_HDUList(self) -> fits.HDUList:
        return fits.HDUList([map_.get_ImageHDU() for map_ in self.maps.values()])
        maps = self.maps.values()
        maps_list = [fits.PrimaryHDU(maps[0])]
        if len(maps) > 1:
            maps_list += [fits.ImageHDU(map_) for map_ in maps]

    @classmethod
    def load(cls, filename):
        file_data = fits.open(filename)
        data = np.array([array.data for array in file_data])
        return cls(data)
    
    def save(self, filename):
        hdu_list = self.get_HDUList
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
        # hdu_list = fits.HDUList([
        #     fits.PrimaryHDU(self.data[0,:,:]),
        #     *[fits.ImageHDU(self.data[i,:,:]) for i in range(1, self.data.shape[0])]
        # ])
        # try:
        #     hdu_list.writeto(filename, overwrite=False)
        # except OSError:
        #     while True:
        #         decision = input(f"{C.RED}{filename} already exists, do you wish to overwrite it? [y/n]{C.END}")
        #         if decision.lower() == "y":
        #             hdu_list.writeto(filename, overwrite=True)
        #             break
        #         elif decision.lower() == "n":
        #             break
