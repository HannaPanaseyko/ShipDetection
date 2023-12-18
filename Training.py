import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os
import pandas as pd
from tensorflow import keras
from keras import utils
from pandas.io.parsers import TextFileReader

from CustomDataGenerator import CustomDataGenerator
from DisplayCallback import DisplayCallback

from PIL import ImageFile
print(tf.__version__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ModelDefinition import create_unet_model
from DataLoading import load_masks_csv

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1150)]
    )
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# path to the dataset folder
IMAGES_PATH = "train_v2"

# path to the csv file with masks
CSV_PATH = "train_ship_segmentations_v2.csv"

# path to the reference image for vusualization of the model performance after each epoch
REFERENCE_IMAGE_NAME = "train_v2/00a52cd2a.jpg"

EPOCHS = 5
BATCH_SIZE = 10
OUTPUT_CLASSES = 1
VALIDATION_SPLIT = 0.001
FILE_LIMIT = None # imageg files limit for loadin
IMAGES_LIMIT = None # max number of images to use (after MASKED_ONLY filter)
MASKED_ONLY = False # use only images with masks
MODEL_FILE = "model768_1.keras"
AUGMENTATION_AMOUNT = 4

def main():
    print("Creating model...")
    model = create_unet_model(OUTPUT_CLASSES)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("Model created")

    utils.plot_model(model, show_shapes=True)

    print("Creating generators...")
    mask_data = load_masks_csv(CSV_PATH)
    train_generator, val_generator = create_generators(IMAGES_PATH, mask_data, BATCH_SIZE, VALIDATION_SPLIT)
    print("Generators created")

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="checkpoint768.keras", monitor="val_accuracy", mode="max", save_weights_only=False, save_best_only=True)

    model_history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[DisplayCallback(model,IMAGES_PATH,  REFERENCE_IMAGE_NAME, mask_data, True), model_checkpoint_callback],
    )
    model.save(MODEL_FILE)

    plot_history(model_history)


def create_generators(dataset_path: str, mask_data: TextFileReader, batch_size: int, validation_split: float):
    file_list = os.listdir(dataset_path)
    image_filenames = [filename for filename in file_list if filename.endswith(".jpg")]
    print(f"Found {len(image_filenames)} images in folder {dataset_path}")


    if FILE_LIMIT is not None:
        image_filenames = image_filenames[:FILE_LIMIT]
        print(f"Using only {FILE_LIMIT} images")

    if MASKED_ONLY:
        image_filenames = [filename for filename in image_filenames if mask_data[mask_data["ImageId"] == filename]["EncodedPixels"].values[0] != ""]
        print(f"Found {len(image_filenames)} images with masks")

    if IMAGES_LIMIT is not None:
        image_filenames = image_filenames[:IMAGES_LIMIT]
        print(f"Using only {IMAGES_LIMIT} images")

    train_generator = CustomDataGenerator(image_filenames, mask_data, batch_size, 1 - validation_split, dataset_path)
    val_generator = CustomDataGenerator(image_filenames, mask_data, batch_size, validation_split, dataset_path)
    
    return train_generator, val_generator


def plot_history(model_history):
    loss = model_history.history["loss"]
    val_loss = model_history.history["val_loss"]

    plt.figure()
    plt.plot(model_history.epoch, loss, "r", label="Training loss")
    plt.plot(model_history.epoch, val_loss, "bo", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
