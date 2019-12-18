import os
from sys import argv

import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from utils import get_lidc_dataframes, load_model_weights, test_example

experiment_name = "5models_13slices_64px"
figs_dir = f"figs/{experiment_name}/"
os.makedirs(figs_dir, exist_ok=True)


def transfer_model(pretrained_model, dropout, fc_layers, num_classes):
    """
    Convert pretrained model to use desired model output layers
    Parameters:
    ---------
        pretrained_model: imagenet trained model with preset weights
        dropout: dropout ratio
        fc_layers: fully connected layer sizes to replace to layers of pretrained model
        num_classes: the number of total outputs

    Returns:
    -------
        a keras model using pretrained model inputs and outputs with class confidences
    """
    for layer in pretrained_model.layers:
        layer.trainable = False

    x = pretrained_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation="relu")(x)
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=pretrained_model.input, outputs=predictions)

    return model


# Plot the training and validation loss + accuracy
def plot_training(hist, i: int):
    """
    Plot training and validation accuracy and loss. Save figures according to global experiment name and model id.

    Parameters:
    ---------
    a keras history object and a model identifier

    Returns:
    -------
        None
    """
    acc = hist.history["acc"]
    val_acc = hist.history["val_acc"]
    epochs = range(len(acc))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Model {i} Training Accuracy")
    plt.savefig(f"figs/{experiment_name}/training_accuracy_m{i}")
    plt.figure()
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(epochs, loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"Model {i} Training Loss")
    plt.savefig(f"figs/{experiment_name}/training_loss_m{i}.png")
    plt.figure()


############### CONSTANT DECLARATIONS #########################
# establish base model image dimensions
HEIGHT, WIDTH = 128, 128
TRAIN_DIR = os.path.realpath("/Users/kclejeune/Downloads/lidc")
if "--data_root" in argv:
    TRAIN_DIR = argv[argv.index("--data_root") + 1]

NUM_SECTORS = 13
if "--num_sectors" in argv:
    NUM_SECTORS = argv[argv.index("--num_sectors") + 1]

# 10 studies per batch
BATCH_SIZE = NUM_SECTORS * 10
NUM_EPOCHS = 10
class_list = ["None", "Benign", "Malignant"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

optimizer = Adam(lr=0.0001)

# collect dataframe buckets
dataframes = get_lidc_dataframes(TRAIN_DIR, NUM_SECTORS)
try:
    num_train_images = sum(len(frame) for frame in dataframes)
except:
    num_train_images = 1012

for i, frame in enumerate(dataframes):
    # instantiate base ResNet50 with imagenet weights and 3 rgb channels
    pretrained_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)
    )

    # construct an image generator with 20% reserved for validation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )

    # construct an image generator for each batch of studies
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=frame,
        # subset="training",
        x_col="File",
        y_col="Label",
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
    )
    # 20% validation data generator
    validation_generator = train_datagen.flow_from_dataframe(
        frame,
        subset="validation",
        x_col="File",
        y_col="Label",
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
    )
    model = transfer_model(
        pretrained_model,
        dropout=dropout,
        fc_layers=FC_LAYERS,
        num_classes=len(class_list),
    )
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    file_dir = os.path.join("checkpoints", "ResNet50", experiment_name)
    os.makedirs(file_dir, exist_ok=True)
    filepath = os.path.join(file_dir, f"_model_weights_{i}.h5")

    # collect testing and validation accuracy and loss for graphing
    history = model.fit_generator(
        train_generator,
        epochs=NUM_EPOCHS,
        workers=8,
        steps_per_epoch=num_train_images // BATCH_SIZE,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=100,
        callbacks=[ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode="max")],
    )
    plot_training(history, i)
