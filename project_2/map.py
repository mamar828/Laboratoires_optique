from __future__ import annotations

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pyregion
from eztcolors import Colors as C


class Array(np.ndarray):
    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        return obj

    def __init__(self, data):
        self.data = data

    def plot(self, filename=None, ax=None, **kwargs):
        if ax:
            plot_func = ax
        else:
            plot_func = plt
            if kwargs.get("figsize"):
                plt.figure(figsize=kwargs.get("figsize"))

        if kwargs.get("discrete_colormap"):
            viridis_cmap = plt.cm.viridis
            cbar_limits = kwargs["cbar_limits"]
            interval = (cbar_limits[1] - cbar_limits[0]) * 2
            bounds = np.linspace(*cbar_limits, interval + 1)
            cmap = ListedColormap(viridis_cmap(np.linspace(0, 1, interval)))
            norm = BoundaryNorm(bounds, cmap.N)
            imshow = plot_func.imshow(self.data, cmap=cmap, norm=norm)
            cbar = plt.colorbar(imshow, ticks=np.linspace(*cbar_limits, interval//2 + 1), fraction=0.046, pad=0.04)

        else:
            imshow = plot_func.imshow(self.data)
            cbar = plt.colorbar(imshow, fraction=0.046, pad=0.04)

        if kwargs.get("cbar_limits") and not kwargs.get("discrete_colormap"):
            imshow.set_clim(*kwargs.get("cbar_limits"))
        if kwargs.get("cbar_label"):
            cbar.set_label(kwargs.get("cbar_label"))
        if kwargs.get("xlabel"):
            plt.xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel"):
            plt.ylabel(kwargs.get("ylabel"))
        
        plot_func.tick_params(axis='both', direction='in')

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=600, bbox_inches="tight")
        elif not ax:
            plt.show()

    def get_ImageHDU(self, name, units):
        return fits.ImageHDU(self.data, fits.Header({"type": self.__class__.__name__, "name": name, "units": units}))


class ValueArray(Array):
    pass


class UncertaintyArray(Array):
    pass


class Map:
    def __init__(
            self,
            value: ValueArray=None,
            uncertainty: UncertaintyArray=None,
            name: str=None,
            units: str=None
    ):
        self.value = value
        self.uncertainty = uncertainty
        self.name = name
        self.units = units

    def __add__(self, other):
        assert (self.units == other.units,
                f"{C.RED+C.BOLD}The two Maps should have the same units to be added.{C.END}")
        self.assert_shapes(other)
        return Map(
            ValueArray(self.value.data + other.value.data),
            UncertaintyArray(self.uncertainty.data + other.uncertainty.data),
            name=f"{self.name} + {other.name}",
            units=self.units
        )

    def __sub__(self, other):
        assert (self.units == other.units,
                f"{C.RED+C.BOLD}The two Maps should have the same units to be subtracted.{C.END}")
        self.assert_shapes(other)
        return Map(
            ValueArray(self.value.data - other.value.data),
            UncertaintyArray(self.uncertainty.data + other.uncertainty.data),
            name=f"{self.name} - {other.name}",
            units=self.units
        )

    def __mul__(self, other):
        self.assert_shapes(other)
        return Map(
            ValueArray(self.value.data * other.value.data),
            UncertaintyArray(((self.uncertainty.data / self.value.data) + (other.uncertainty.data / other.value.data)) \
             * self.value.data * other.value.data),
            name=f"{self.name} * {other.name}",
            units=f"{self.units} * {other.units}"
        )
    
    def __truediv__(self, other):
        self.assert_shapes(other)
        return Map(
            ValueArray(self.value.data / other.value.data),
            UncertaintyArray(((self.uncertainty.data / self.value.data) + (other.uncertainty.data / other.value.data)) \
             * self.value.data / other.value.data),
            name=f"{self.name} / {other.name}",
            units=f"{self.units} / {other.units}"
        )
    
    def __getitem__(self, key: slice) -> np.ndarray:
        return np.stack([self.value[key], self.uncertainty[key]], axis=1)
    
    @property
    def shape(self):
        return self.value.data.shape

    @classmethod
    def load(cls, filename):
        hdu_list = fits.open(filename)
        value, uncertainty = None, None
        for element in hdu_list:
            if element.header["type"] == "ValueArray":
                value = ValueArray(element.data)
            if element.header["type"] == "UncertaintyArray":
                uncertainty = UncertaintyArray(element.data)
        return cls(value, uncertainty, hdu_list[0].header["name"], hdu_list[0].header["units"])
    
    def save(self, filename):
        hdu_list = fits.HDUList([])
        if self.value is not None:
            hdu_list.append(self.value.get_ImageHDU(self.name, self.units))
        if self.uncertainty is not None:
            hdu_list.append(self.uncertainty.get_ImageHDU(self.name, self.units))
        
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

    def assert_shapes(self, other):
        assert (self.shape == other.shape,
                f"{C.RED+C.BOLD}Both Maps should have the same shapes. Current are {self.shape} and {other.shape}.")
        
    def get_masked_region(self, region: pyregion.core.ShapeList) -> Map:
        mask = region.get_mask(fits.PrimaryHDU(self.value.data))
        mask = np.where(mask == False, np.nan, 1)
        return Map(
            self.value * mask,
            self.uncertainty * mask,
            self.name,
            self.units
        )
