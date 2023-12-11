import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

from ImageMaskLoading import rle_to_mask

CSV_PATH = "train_ship_segmentations_v2.csv"
IMAGES_PATH = "train_v2"
MODEL_FILE = "./model.keras"

model = tf.keras.models.load_model(MODEL_FILE) 
types = {"ImageId": "str", "EncodedPixels": "str"}
mask_data = pd.read_csv(CSV_PATH, dtype=types, keep_default_na=False)

input_image = cv2.imread('{DATA_PATH}/000155de5.jpg')
input_image = cv2.resize(input_image, (768, 768))  # adjust dimensions
input_image = input_image / 255.0  # normalize if needed

mask_info = mask_data[mask_data['ImageId'] == '000155de5.jpg']['EncodedPixels'].values
combined_mask = ' '.join(mask_info)
mask = rle_to_mask(combined_mask)
rgb_mask = np.tile(mask, (1, 1, 3))

# Predict the mask
predicted_mask = model.predict(np.expand_dims(input_image, axis=0))
# Post-process the mask (adjust threshold based on your application)

#threshold = 0
binary_mask = predicted_mask.astype(np.float32)

# Visualize or save the results as needed
cv2.imshow('Input Image', input_image)
cv2.imshow('Predicted Mask', predicted_mask[0, :, :, :])
cv2.imshow('Expected mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()