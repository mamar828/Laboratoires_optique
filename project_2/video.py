import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import gzip
import pickle
from eztcolors import Colors as C
import multiprocessing
from tqdm import tqdm
from time import time

from project_2.spectrum import Spectrum
from project_2.map import *



class Video:
    def __init__(self, filename):
        self.filename = filename
        if filename.split(".")[-1] == "MP4":
            self.data = self.get_numpy_array(filename)
        elif filename.split(".")[-1] == "npy":
            self.data = np.load(filename)
        else:
            print(f"{C.RED}Error: {filename} is not a valid file.{C.END}")

    def get_numpy_array(self, filename):
        capture = cv2.VideoCapture(filename)
        frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        data = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        frame_number = 0

        while frame_number < frameCount:
            booli, frame = capture.read()
            if frame is not None:
                data[frame_number] = frame
            else:
                break
            frame_number += 1

        return data[:frame_number,:,:,:]

    def save(self):
        with gzip.open(f"{self.filename[:-4]}.gz", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with gzip.open(filename, "rb") as file:
            return pickle.load(file)
    
    def plot(self):
        for i in range(self.data.shape[0]):
            plt.imshow(self.data[i,:,:], cmap="magma")
            plt.show()

    def crop(self, x_slice, y_slice):
        self.data = self.data[:, y_slice, x_slice]
        print(f"New dimensions: {tuple(reversed(self.data.shape[1:3]))}")
        with open(f"project_2/crop_slices/{self.filename.split('/')[-1].split('.')[0]}.txt", "w") as f:
            f.write(self.filename)
            f.write(f"\n\tx_slice: {x_slice}, \n\ty_slice: {y_slice}")

    def convert_to_luminosity(self):
        self.data = np.mean(self.data / 255, axis=3)
    
    def bin(self, bins: tuple[int,int,int]) -> np.ndarray:
        # bins are given in x, y, z order, but the shape is in z, y, x order
        r_bins = list(reversed(bins))
        cropped_pixels = np.array(self.data.shape[:3]) % np.array(r_bins)
        self.data = self.data[:self.data.shape[0] - cropped_pixels[0], :self.data.shape[1] - cropped_pixels[1], 
                         :self.data.shape[2] - cropped_pixels[2], ...]
        if len(self.data.shape) == 4:
            dtype = np.int64
        else:
            dtype = np.float64

        for ax, b in enumerate(r_bins):
            if b != 1:
                indices = list(self.data.shape)
                indices[ax:ax+1] = [int(self.data.shape[ax]/b), b]
                reshaped_data = self.data.reshape(indices)
                self.data = reshaped_data.mean(axis=ax+1, dtype=dtype)

    def get_spectrum(self, x: int, y: int, z_limit: int=None):
        assert len(self.data.shape) == 3, (
                f"{C.RED+C.BOLD}Video must be converted to luminosity before creating a spectrum{C.END}")
        data = np.stack((np.arange(self.data.shape[0]), self.data[:,y,x]), axis=1)
        if z_limit:
            return Spectrum(data[:z_limit,:])
        else:
            return Spectrum(data)
    
    def fit(self) -> tuple[Map, Map]:
        start = time()
        data = self.data[:155,:,:]
        with multiprocessing.Pool() as pool:
            results = []
            mapped_pool = pool.imap(worker_fit, [data[:,y,:] for y in range(data.shape[1])])
            print(C.LIGHT_BLUE, end="")
            for result in tqdm(mapped_pool, total=data.shape[1], desc="Fitting"):
                results.append(result)
            print(C.END, end="")

        results_array = np.array(results)
        print(f"{C.GREEN+C.BOLD}Finished fitting in {time()-start} seconds.{C.END}")
        output = (
            Map(value=ValueArray(results_array[:,:,0]),
                uncertainty=UncertaintyArray(results_array[:,:,1]), 
                name="FWHM",
                units="frames"),
            Map(value=ValueArray(results_array[:,:,2]),
                name="FWHM standard deviation",
                units="frames")
        )
        return output
    
    def measure_peak_distances(self) -> Map:
        start = time()
        results = []
        spectral_axis = np.arange(self.data.shape[0])
        print(C.LIGHT_BLUE, end="")
        for y in tqdm(range(self.data.shape[1]), desc="Measuring peaks"):
            for x in range(self.data.shape[2]):
                spec = Spectrum(np.stack((spectral_axis, self.data[:,y,x]), axis=1))
                results.append(spec.get_peak_distance())
        print(C.END, end="")

        values = np.array(results).reshape(self.data.shape[1:])
        uncertainties = np.zeros_like(values) + 2
        print(f"{C.GREEN+C.BOLD}Finished measuring peaks in {time()-start} seconds.{C.END}")
        output = Map(
            value=ValueArray(values),
            uncertainty=UncertaintyArray(uncertainties),
            name="average peak distance",
            units="frames"
        )
        return output


def worker_fit(data) -> np.ndarray:
    line = []
    spectral_axis = np.arange(data.shape[0])
    for x in range(data.shape[1]):
        spec = Spectrum(np.stack((spectral_axis, data[:,x]), axis=1))
        spec.fit()
        line.append([*spec.get_FWHM(), spec.get_residue_stddev()])
    return line
