import cv2
import numpy as np
import pandas as pd
import keras

from ImageMaskLoading import rle_to_mask

CSV_PATH = "train_ship_segmentations_v2.csv"
IMAGES_PATH = "train_v2"
MODEL_FILE = "./checkpoint.keras"

model = keras.models.load_model(MODEL_FILE) 
types = {"ImageId": "str", "EncodedPixels": "str"}
mask_data = pd.read_csv(CSV_PATH, dtype=types, keep_default_na=False)

input_image_cv = cv2.imread(IMAGES_PATH + "/0a1a7f395.jpg")
input_image = keras.utils.img_to_array(keras.utils.load_img(IMAGES_PATH +"/0a1a7f395.jpg", target_size=(128, 128))) / 255.0

mask_info = mask_data[mask_data['ImageId'] == '000155de5.jpg']['EncodedPixels'].values
combined_mask = ' '.join(mask_info)
mask = rle_to_mask(combined_mask)

# Predict the mask
predicted_mask = model.predict(np.expand_dims(input_image, axis=0))
# Post-process the mask (adjust threshold based on your application)

#threshold = 0
predicted_mask_uint8 = (predicted_mask[0] * 255).astype(np.uint8)

# Visualize or save the results as needed
cv2.imshow('Input Image', input_image_cv)
cv2.imshow('Predicted Mask', predicted_mask_uint8)
cv2.imshow('Expected mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()