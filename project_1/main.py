import matplotlib.pyplot as plt

from video import Video


def plot(array, save_filename=None):
    plot = plt.colorbar(plt.imshow(array, cmap="hot"))
    plt.axis("off")
    
    plt.tight_layout()
    plot.ax.set_ylabel("Motion energy [-]")
    if save_filename:
        plt.savefig(save_filename, dpi=600)
    else:
        plt.show()


def extract_videos(video_name):
    vr = Video(f"project_1/video_data/{video_name}.MP4")
    vr.crop(slice(375+125, 1125+125), slice(50, 800))
    # vr.plot()
    vr.convert_to_luminosity()

    # vr = Video(f"project_1/arrays/{video_name}.npy")
    motion_energy = vr.get_mean_motion_energy()
    plot(motion_energy, f"project_1/figures/{video_name}.png")


extract_videos("upwards_plane")
