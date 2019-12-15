import os
import convolutional
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
import glob
from os.path import dirname, abspath, join
from random import shuffle

NUM_EPOCHS = 5
RETRAIN = True

# TODO: define train set, test set, and validation set from input data
def main():
    current_folder = dirname(abspath(__file__))
    data_folder = join(current_folder, "imagenet_images")
    model_folder = join(current_folder, "saved_models")
    accel_folder = join(data_folder, "accelerator")
    bell_folder = join(data_folder, "belladonna")
    image_set = []
    for filename in glob.glob(os.path.join(accel_folder, "*.jpg")):
        img = Image.open(filename)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)
        image_set.append((img, 1))

    for filename in glob.glob(os.path.join(bell_folder, "*.jpg")):
        img = Image.open(filename)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)
        image_set.append((img, 0))
    shuffle(image_set)
    training_num = int(len(image_set) * 0.75)
    training_set = image_set[:training_num]
    testing_set = image_set[training_num:]

    train_loader, test_loader, val_loader = convolutional.get_loaders(
        training_set, testing_set
    )

    net = convolutional.Net()
    if RETRAIN:
        convolutional.train(net, NUM_EPOCHS, train_loader, val_loader)
        convolutional.test(net, test_loader)
    else:
        net.load_state_dict(torch.load(join(model_folder, "model")))
    convolutional.test(net, test_loader)

    # torch.save(net.state_dict(), join(model_folder, "model"))


if __name__ == "__main__":
    main()
