import os

import convolutional
import torch
import torch.multiprocessing as mp
from PIL import Image
import glob
from os.path import dirname, abspath, join


# TODO: define train set, test set, and validation set from input data
def main():

    image_set = []

    current_folder = dirname(abspath(__file__))
    data_folder = join(current_folder, "imagenet_images")
    accel_folder = join(data_folder, "accelerator")

    for filename in glob.glob(os.path.join(accel_folder, '*.jpg')):
        image_set.append(Image.open(filename))

    training_set = image_set[:10]
    testing_set = image_set[10:12]

    train_loader, test_loader, val_loader = convolutional.get_loaders(training_set, testing_set)

    net = convolutional.Net()

    convolutional.train(net, 100, training_set, testing_set)

    pass


if __name__ == '__main__':
    main()