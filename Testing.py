import numpy as np
import keras
from matplotlib import pyplot as plt 

from DataLoading import load_masks_csv, read_image, get_mask

CSV_PATH = "/home/slava/dev/ShipDetection/train_ship_segmentations_v2.csv"
IMAGES_PATH = "/home/slava/dev/ShipDetection/train_v2"
MODEL_FILE = "./checkpoint768.keras"
IMAGE_NAME = "00a52cd2a.jpg"

model = keras.models.load_model(MODEL_FILE)

mask_data = load_masks_csv(CSV_PATH)

input_image = read_image(IMAGES_PATH, IMAGE_NAME)

mask = get_mask(mask_data, IMAGE_NAME)

# Predict the mask
predicted_mask = model.predict(np.expand_dims(input_image, axis=0))
predicted_mask_uint8 = (predicted_mask[0] * 255).astype(np.uint8)

# plot image and expected mask
plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask_uint8)
plt.show()
