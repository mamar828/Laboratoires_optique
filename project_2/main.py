import matplotlib.pyplot as plt
import numpy as np
from time import time

from project_2.video import Video
from project_2.map import Map


def plot(array, save_filename=None):
    plt.imshow(array, cmap="hot")
    plt.axis("off")
    
    plt.tight_layout()
    plt.colorbar().ax.set_ylabel("Motion energy [-]")
    if save_filename:
        plt.savefig(save_filename, dpi=600)
    else:
        plt.show()
    plt.clf()


def extract_videos(video_name):
    # video = Video(f"project_2/video_data/{video_name}.MP4")
    # video.crop(slice(680, 1365), slice(215, 900))
    # video.convert_to_luminosity()
    # video.save()

    video = Video.load(f"project_2/video_data/{video_name}.gz")

    # video.bin(bins=(1,1,1))
    # video.data = video.data[:,375:383,375:380]
    # video.data = video.data[:,375:425,375:425]
    # video.plot()
    fitted_map = video.fit()
    fitted_map.save("project_2/maps/fit_results.fits")
    fitted_map.plot()
    fitted_map.plot(save_filename="project_2/maps/FWHM.png")


if __name__ == "__main__":
    extract_videos("MVI_9147")
