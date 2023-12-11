from tensorflow import keras
import os
import numpy as np
from ImageMaskLoading import load_image_and_mask
import keras
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, mask_data, batch_size, validation_split, dataset_path):
        self.image_filenames = image_filenames
        self.mask_data = mask_data
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_samples = len(image_filenames)
        self.num_validation_samples = int(validation_split * self.num_samples)
        self.mode = "train"
        self.dataset_path = dataset_path

        self.image_data_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest"
        )

    def __len__(self):
        # Return the number of batches per epoch
        if self.mode == "train":
            return (self.num_samples - self.num_validation_samples) // self.batch_size
        else:
            return self.num_validation_samples // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        if self.mode == "train":
            start_idx = index * self.batch_size
            end_idx = (index + 1) * self.batch_size
            batch_files = self.image_filenames[
                          self.num_validation_samples + start_idx: self.num_validation_samples + end_idx]
        else:
            start_idx = index * self.batch_size
            end_idx = (index + 1) * self.batch_size
            batch_files = self.image_filenames[start_idx: start_idx + self.batch_size]

        batch_images = []
        batch_masks = []
        for image_filename in batch_files:
            image_path = os.path.join(self.dataset_path, image_filename)
            image, mask = load_image_and_mask(image_path, self.mask_data)
            batch_images.append(image)
            batch_masks.append(mask)
            for _ in range(4):
                self.__add_random_transform(batch_images, batch_masks, image, mask)

        return np.array(batch_images), np.array(batch_masks)

    def __add_random_transform(self, batch_images: list, batch_masks: list, image, mask):
        transform_parameters = self.image_data_generator.get_random_transform(image.shape)
        transformed_imge = self.image_data_generator.apply_transform(image, transform_parameters)
        transformed_mask = self.image_data_generator.apply_transform(mask, transform_parameters)
        batch_images.append(transformed_imge)
        batch_masks.append(transformed_mask)

    def on_epoch_end(self):
        # Shuffle the data after each epoch
        np.random.shuffle(self.image_filenames)

    def set_mode(self, mode):
        self.mode = mode
