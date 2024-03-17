import numpy as np
import cv2
import matplotlib.pyplot as plt

from eztcolors import Colors as C


class Video:
    def __init__(self, filename, save_foldername="project_1/arrays"):
        self.filename = filename
        self.save_foldername = save_foldername
        if filename.split(".")[-1] == "MP4":
            self.data = self.get_numpy_array(filename)
        elif filename.split(".")[-1] == "npy":
            self.data = np.load(filename)
        else:
            print(f"{C.RED}Error: {filename} is not a valid file.{C.END}")

    def get_numpy_array(self, filename):
        capture = cv2.VideoCapture(filename)
        frameCount = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 10
        frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        data = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        frame_number = 0

        while frame_number < frameCount:
            booli, frame = capture.read()
            if frame is not None:
                data[frame_number] = frame
            frame_number += 1
        
        return data

    def save(self):
        np.save(f"{self.save_foldername}/{self.filename.split('/')[-1].split('.')[0]}.npy", self.data)
        
    def plot(self):
        for i in range(self.data.shape[0]):
            plt.imshow(self.data[i,:,:])
            plt.show()

    def crop(self, x_slice, y_slice):
        self.data = self.data[:, y_slice, x_slice]
        print(f"New dimensions: {self.data.shape[1:3]}")
        with open(f"project_1/crop_slices/{self.filename.split('/')[-1].split('.')[0]}.txt", "w") as f:
            f.write(self.filename)
            f.write(f"\n\tx_slice: {x_slice}, \n\ty_slice: {y_slice}")

    def convert_to_luminosity(self):
        self.data = np.mean(self.data / 255, axis=3)
    
    def get_mean_motion_energy(self) -> np.ndarray:
        differences = np.abs(np.diff(self.data, axis=0))
        mean = np.mean(differences, axis=0)
        return mean
