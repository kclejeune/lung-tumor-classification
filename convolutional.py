import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data as data
import torch.utils.data.sampler as sam
import torch

LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_TRAINING_SAMPLES = 20
NUM_TESTING_SAMPLES = 20
NUM_VAL_SAMPLES = 20

train_sampler = sam.SubsetRandomSampler(np.arange(NUM_TRAINING_SAMPLES, dtype=np.int64))
test_sampler = sam.SubsetRandomSampler(np.arange(NUM_TESTING_SAMPLES, dtype=np.int64))
val_sampler = sam.SubsetRandomSampler(np.arange(NUM_TRAINING_SAMPLES, NUM_TRAINING_SAMPLES + NUM_VAL_SAMPLES,
                                                dtype=np.int64))


# net with 2 convolutions, and a binary output
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.MaxPool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_opt_and_loss(model):
    optimizer = opt.Adam(filter((lambda p: p.requires_grad, model.parameters())), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()
    return optimizer, loss


def get_loaders(train_set, test_set):
    train_loader = data.DataLoader(train_set, batch_size = BATCH_SIZE, sampler=train_sampler, num_workers=2)
    test_loader = data.DataLoader(test_set, batch_size = BATCH_SIZE, sampler=test_sampler, num_workers=2)
    val_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
    return train_loader, test_loader, val_loader


def train(net, epochs, train_loader, val_loader):
    print("Training for " + str(epochs) + " epochs...")
    optim, loss = get_opt_and_loss(net)

    total_loss = 0
    total_val_loss = 0
    for epoch in range(epochs):
        running_loss = 0
        running_val_loss = 0

        for i, item in enumerate(train_loader, 0):
            inputs, labels = item
            optim.zero_grad()
            outputs = net(inputs)
            current_loss = loss(outputs, labels)
            current_loss.backward()
            optim.step()
            running_loss += current_loss.item()
            total_loss += current_loss.item()

        print("Epoch " + str(epoch) + " loss: " + str(running_loss))

        for i, item in enumerate(val_loader, 0):
            inputs, labels = item
            outputs = net(inputs)
            current_loss = loss(outputs, labels)
            running_val_loss += current_loss.item()
            total_val_loss += current_loss.item()

        print("Epoch " + str(epoch) + " val loss: " + str(running_val_loss))


# this function will only take in a model which has the convolutional layer weights
# transferred in from the ImageNet classifier
# i think this should work but definitely needs testing
def train_last_two(net, epochs, train_loader, val_loader):
    net.conv1.weight.requires_grad = False
    net.conv1.bias.requires_grad = False
    net.conv2.weight.requires_grad = False
    net.conv2.bias.requires_grad = False

    train(net, epochs, train_loader, val_loader)


def test(net, test_loader):
    correct = 0
    for i, item in enumerate(test_loader):
        inputs, labels = item
        outputs = net(inputs).squeeze()

        _, pred = torch.max(outputs, 1)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    acc = correct/len(test_loader)
    print("Accuracy: " + str(acc))














