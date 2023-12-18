import numpy as np
import keras
import pandas as pd

img_height = 768
img_width = 768

def rle_to_mask(rle, shape = (img_height, img_width, 1)) -> np.ndarray:
    array = np.zeros(shape, dtype=float).flatten()
    splitted = split_rle(rle)
    for (start, length) in splitted:
        for i in range(start - 1, start - 1 + length):
            array[i] = 1

    array = array.reshape((img_height, img_width, 1))

    rotated= np.rot90(array, 3)
    mirrored = np.fliplr(rotated)
    # invert data
    #mirrored = 1 - mirrored
    return mirrored

def split_rle(rle_string) -> list:
    # Split the RLE string into individual characters
    rle_chars = rle_string.split()

    # Initialize an empty list to store pairs
    pairs = []

    # Iterate through the characters
    i = 0
    while i < len(rle_chars):
        # Extract the count and value from consecutive characters
        count = int(rle_chars[i])
        value = int(rle_chars[i + 1])
        pairs.append((count, value))

        # Move the index to the next set of characters
        i += 2

    return pairs

def load_image_and_mask(images_path, image_name, mask_data) -> (np.ndarray, np.ndarray):
    image = read_image(images_path, image_name)
    mask = get_mask(mask_data, image_name)
    
    return image, mask

def get_mask(mask_data, image_name) -> np.ndarray:
    mask_info = mask_data[mask_data['ImageId'] == image_name]['EncodedPixels'].values
    mask = rle_to_mask(mask_info[0])
    return mask


def load_masks_csv(CSV_PATH) -> pd.DataFrame:
    types = {"ImageId": "str", "EncodedPixels": "str"}
    mask_data = pd.read_csv(CSV_PATH, dtype=types, keep_default_na=False)
    return mask_data


def read_image(images_path, image_name) -> np.ndarray:
    return keras.utils.img_to_array(keras.utils.load_img(images_path +"/" + image_name, target_size=(768, 768))) / 255.0
