from scipy import ndimage
from PIL.Image import open as open_img
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import os
from os.path import realpath, join
from typing import List
import platform

if "--image_root" in argv:
    i = argv.index("--image_root") + 1
    image_dir = os.path.abspath(argv[i])
else:
    keybase_base = (
        os.path.realpath(":k")
        if platform.system() == "Windows"
        else os.path.realpath("/keybase")
    )
    image_dir = os.path.join(keybase_base, "team", "cwru_dl", "imagenet")


class Image:
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __eq__(self, other):
        return (
            isinstance(other, ImageClass)
            and other.image == self.image
            and other.label == self.label
        )


def parse_numeric_images(image_root: str = image_dir, grayscale=True):
    """
    Parameters
    ---------
        image_root: the directory to find the image class folders
        grayscale: whether to return grayscale images
    Return
    ------
        returns a numpy array of Image objects with attributes image and label. image.shape = (w, h)
        with dimensions equivalent to the image size, and each element value is set according
        to the relative luminosity via

        https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.convert
    """
    image_arr = []
    for class_label in os.scandir(image_root):
        path = os.path.join(image_root, class_label)
        # print(f"Class {class_label} Path {path}")
        for img in os.scandir(path):
            img_path = os.path.join(path, img)
            try:
                image = np.asarray(
                    open_img(img_path).convert("L") if grayscale else open_img(img_path)
                )
                # print(tuple((image.shape, class_label)))
                image_arr.append(Image(image=image, label=class_label))
            except IOError:
                print(f"Unable to read file {img_path}")
    return np.array(image_arr)


imgs = parse_numeric_images()

# example: get the last image added and show it
img = imgs[-1]
print(img.image.shape)
plt.imshow(img.image)
plt.show()
