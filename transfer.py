import os
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from utils import get_lidc_dataframes

TRAIN = True


experiment_name = "5models_13slices_128px"
figs_dir = f"figs/{experiment_name}/"
os.makedirs(figs_dir, exist_ok=True)


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


# Plot the training and validation loss + accuracy
def plot_training(hist, i):
    acc = hist.history["acc"]
    # val_acc = hist.history["val_acc"]
    epochs = range(len(acc))
    plt.plot(epochs, acc, label="Training Accuracy")
    # plt.plot(epochs, val_acc, label="Validation Accuracy")
    # plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Model {i} Training Accuracy")
    plt.savefig(f"figs/{experiment_name}/training_accuracy_m{i}")
    plt.figure()
    loss = hist.history["loss"]
    # val_loss = hist.history["val_loss"]
    plt.plot(epochs, loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"Model {i} Training Loss")
    plt.savefig(f"figs/{experiment_name}/training_loss_m{i}.png")
    plt.figure()


############### CONSTANT DECLARATIONS #########################
# establish base model image dimensions
HEIGHT, WIDTH = 128, 128
TRAIN_DIR = os.path.realpath("/Users/kclejeune/Downloads/lidc")
NUM_SLICES = 5
# 10 studies per batch
BATCH_SIZE = NUM_SLICES * 10
NUM_EPOCHS = 10
class_list = ["None", "Benign", "Malignant"]
FC_LAYERS = [1024, 1024]
dropout = 0.5
num_train_images = 1012
optimizer = Adam(lr=0.0001)
# this is returning the same number of datasets as the number of slices
dataframes = get_lidc_dataframes(TRAIN_DIR, NUM_SLICES)
print("Num Slices:", NUM_SLICES)
print("Num Models:", len(dataframes))
if TRAIN:
    for i, frame in enumerate(dataframes):
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3)
        )

        # construct an image generator with 20% reserved for validation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            # validation_split=0.2,
        )
        # construct an image generator for each batch of studies
        train_generator = train_datagen.flow_from_dataframe(
            frame,
            # subset="training",
            x_col="File",
            y_col="Label",
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
        )
        # # 20% validation data generator
        # validation_generator = train_datagen.flow_from_dataframe(
        #     frame,
        #     subset="validation",
        #     x_col="File",
        #     y_col="Label",
        #     target_size=(HEIGHT, WIDTH),
        #     batch_size=BATCH_SIZE,
        # )
        finetune_model = build_finetune_model(
            base_model,
            dropout=dropout,
            fc_layers=FC_LAYERS,
            num_classes=len(class_list),
        )
        finetune_model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        file_dir = os.path.join("checkpoints", "ResNet50", experiment_name)
        os.makedirs(file_dir, exist_ok=True)
        filepath = os.path.join(file_dir, f"_model_weights_{i}.h5")
        history = finetune_model.fit_generator(
            train_generator,
            epochs=NUM_EPOCHS,
            workers=8,
            steps_per_epoch=num_train_images // BATCH_SIZE,
            shuffle=True,
            # validation_data=validation_generator,
            # validation_steps=100,
            callbacks=[
                ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode="max")
            ],
        )
        plot_training(history, i)

