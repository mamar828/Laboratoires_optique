import matplotlib.pyplot as plt
import os

from project_1.video import Video


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
    video = Video(f"project_1/video_data/{video_name}.MP4")
    video.crop(slice(520-50, 1345-50), slice(5, 860))
    # video.crop(slice(520-40, 1345-40), slice(5, 860))
    # video.crop(slice(520, 1345), slice(5, 860))
    # video.plot()
    video.convert_to_luminosity()

    # video = Video(f"project_1/arrays/{video_name}.npy")
    motion_energy = video.get_mean_motion_energy()
    plot(motion_energy, f"project_1/figures/{video_name}.png")


extract_videos("vide")


def autoextract_videos():
    for video_name in os.listdir("project_1/video_data"):
        print(video_name, end=" "*(40-len(video_name)))
        video = Video(f"project_1/video_data/{video_name}")
        with open(f"project_1/crop_slices/{video_name.split('.')[0]}.txt", "r") as f:
            lines = f.read().split("\n")[1:]
            x_slice = slice(int(lines[0].split("(")[1].split(",")[0]), int(lines[0].split(", ")[1].split(",")[0]))
            y_slice = slice(int(lines[1].split("(")[1].split(",")[0]), int(lines[1].split(", ")[1].split(",")[0]))
        video.crop(x_slice, y_slice)
        video.convert_to_luminosity()

        motion_energy = video.get_mean_motion_energy()
        plot(motion_energy, f"project_1/figures/{video_name.split('.')[0]}.png")

# autoextract_videos()
