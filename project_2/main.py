import matplotlib.pyplot as plt
import numpy as np
from time import time
import pyregion

from project_2.video import Video
from project_2.map import *


def plot(array, filename=None):
    plt.imshow(array, cmap="hot")
    plt.axis("off")
    
    plt.tight_layout()
    plt.colorbar().ax.set_ylabel("Motion energy [-]")
    if filename:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()
    plt.clf()


def extract_video(video_name):
    video = Video(f"project_2/video_data/{video_name}.MP4")
    video.crop(slice(680, 1365), slice(215, 900))
    video.convert_to_luminosity()
    video.save()


# extract_video("MVI_9147")


def fit_video(video_name):
    video = Video.load(f"project_2/video_data/{video_name}.gz")
    fwhm, fwhm_std = video.fit()
    fwhm.save("project_2/maps/fwhm.fits")
    fwhm_std.save("project_2/maps/fwhm_std.fits")
    fwhm.value.plot(filename="project_2/maps/FWHM.png")
    

# if __name__ == "__main__":
#     fit_video("MVI_9147")


def measure_peaks(video_name):
    video = Video.load(f"project_2/video_data/{video_name}.gz")
    peak_dists = video.measure_peak_distances()
    peak_dists.save("project_2/maps/peak_dists.fits")
    peak_dists.value.plot(filename="project_2/maps/peak_dists.png")


# measure_peaks("MVI_9147")


def get_finesse():
    fwhm = Map.load("project_2/maps/fwhm.fits")
    peak_dists = Map.load("project_2/maps/peak_dists.fits")
    finesse = peak_dists / fwhm
    finesse.name = "finesse"
    finesse.units = "-"
    finesse.save("project_2/maps/finesse.fits")
    finesse.value.plot("project_2/maps/finesse.png")


# get_finesse()


def make_spectrum_graph():
    video = Video.load("project_2/video_data/MVI_9147.gz")
    spectrums = [
        video.get_spectrum(110, 220),
        video.get_spectrum(110, 220, 155),
        video.get_spectrum(475, 485),
        video.get_spectrum(475, 485, 155)
    ]
    spectrums[1].fit()
    print(spectrums[1].get_r_squared())
    spectrums[3].fit()
    print(spectrums[3].get_r_squared())
    fig, axs = plt.subplots(4, 1, figsize=(4,7))
    for spectrum, ax, letter in zip(spectrums, axs, ["a)", "b)", "c)", "d)"]):
        spectrum.plot(ax=ax)
        ax.set_title(letter, loc="left")
        ax.tick_params(axis='both', direction='in')
    
    fig.supxlabel("Image vidéo [-]")
    fig.supylabel("Intensité normalisée [-]")
    plt.tight_layout()
    # plt.savefig("project_2/figures/spectrums.png", dpi=600, bbox_inches="tight")
    # plt.show()


# make_spectrum_graph()


def make_fwhm_peak_dists_graph():
    fwhm = Map.load("project_2/maps/fwhm.fits")
    peak_dists = Map.load("project_2/maps/peak_dists.fits")

    fig, axs = plt.subplots(2, 1, figsize=(5, 7))
    fwhm.value.plot(ax=axs[0], cbar_label="Largeur à mi-hauteur [u. arb.]", cbar_limits=(10,50))
    peak_dists.value.plot(ax=axs[1], cbar_label="Distance moyenne entre les pics [u. arb.]", cbar_limits=(150,157),
                          discrete_colormap=True)

    axs[0].set_title("a)", loc="left")
    axs[1].set_title("b)", loc="left")
    fig.supxlabel("Position x [pixels]")
    fig.supylabel("Position y [pixels]")
    plt.tight_layout()
    plt.savefig("project_2/figures/fwhm_peak_dists.png", dpi=600, bbox_inches="tight")
    # plt.show()


# make_fwhm_peak_dists_graph()


def make_finesse_graph():
    finesse = Map.load("project_2/maps/finesse.fits")

    fig, axs = plt.subplots(2, 1, figsize=(5, 7))

    finesse.value.plot(
        ax=axs[0],
        cbar_label="Finesse [-]",
        cbar_limits=(3,15)
    )
    axs[0].set_xlabel("Position x [pixels]")
    axs[0].set_ylabel("Position y [pixels]")

    array = finesse[365:650,348]
    x = range(array.shape[0])
    axs[1].errorbar(x, array[:,0], yerr=array[:,1], markersize=2, fmt="ko")
    axs[1].set_xlabel("Distance par rapport au centre [pixels]")
    axs[1].set_ylabel("Finesse [-]")

    axs[0].set_title("a)", loc="left")
    axs[1].set_title("b)", loc="left")
    plt.tight_layout()
    plt.savefig("project_2/figures/finesse.png", dpi=600, bbox_inches="tight")
    # plt.show()


# make_finesse_graph()


def get_stats():
    finesse = Map.load("project_2/maps/finesse.fits")
    masked_finesse = finesse.get_masked_region(pyregion.open("project_2/regions/big_circle.reg"))
    max_i = np.unravel_index(np.nanargmax(masked_finesse.value), shape=masked_finesse.shape)
    min_i = np.unravel_index(np.nanargmin(masked_finesse.value), shape=masked_finesse.shape)
    print(f"max finesse : {masked_finesse.value[max_i]} ± {masked_finesse.uncertainty[max_i]} at pixel {max_i}")
    print(f"min finesse : {masked_finesse.value[min_i]} ± {masked_finesse.uncertainty[min_i]} at pixel {min_i}")

    masked_finesse = finesse.get_masked_region(pyregion.open("project_2/regions/small_circle.reg"))
    max_i = np.unravel_index(np.nanargmax(masked_finesse.value), shape=masked_finesse.shape)
    print(f"middle max finesse : {masked_finesse.value[max_i]} ± {masked_finesse.uncertainty[max_i]} at pixel {max_i}")


get_stats()
