import os

import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from utils import get_keybase_team, get_lidc_dataframe

# establish base model
HEIGHT, WIDTH = 512, 512

base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)
)

TRAIN_DIR = os.path.join(get_keybase_team("cwru_dl"), "lidc")
BATCH_SIZE = 8

lidc_df = get_lidc_dataframe(TRAIN_DIR)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
)

train_generator = train_datagen.flow_from_dataframe(
    lidc_df,
    x_col="File",
    y_col="Label",
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
)


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation="relu")(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation="softmax")(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


class_list = ["None", "Malignant", "Benign"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(
    base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list)
)


NUM_EPOCHS = 10
BATCH_SIZE = 8
num_train_images = 10000

adam = Adam(lr=0.0001)
finetune_model.compile(adam, loss="categorical_crossentropy", metrics=["accuracy"])

filepath = os.path.join("checkpoints", "ResNet50", "_model_weights.h5")
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode="max")
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(
    train_generator,
    epochs=NUM_EPOCHS,
    workers=8,
    steps_per_epoch=num_train_images // BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks_list,
)


# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.plot(epochs, acc, "r.")
    plt.plot(epochs, val_acc, "r")
    plt.title("Training and validation accuracy")

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig("acc_vs_epochs.png")


plot_training(history)
