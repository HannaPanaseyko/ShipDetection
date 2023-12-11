import numpy as np
import keras
import os
import cv2

img_height = 768
img_width = 768

def rle_to_mask(rle, shape = (img_height, img_width, 1)):
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

def split_rle(rle_string):
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

def load_image_and_mask(image_path, mask_data):
    image = keras.utils.img_to_array(keras.utils.load_img(image_path, target_size=(128, 128))) / 255.0
    image_name = os.path.basename(image_path)
    
    # Find the corresponding mask data in the CSV file
    mask_info = mask_data[mask_data['ImageId'] == image_name]['EncodedPixels'].values

    mask = rle_to_mask(mask_info[0])

    
    # Convert mask from grayscale to RGB
   # mask_rgb = np.repeat(mask, 3, axis=2)

    #downscaled_image = cv2.resize(image, (128, 128),  interpolation=cv2.INTER_AREA)
    downscaled_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
    # (128, 128) -> (128, 128, 1)
    #downscaled_mask = np.expand_dims(mask, axis=2)
    #downscaled_mask = np.repeat(downscaled_mask, 3, axis=2)
    downscaled_mask = downscaled_mask.reshape((128, 128, 1))
    
    return image, downscaled_mask
