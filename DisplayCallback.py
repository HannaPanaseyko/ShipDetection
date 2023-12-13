from ImageMaskLoading import rle_to_mask
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import keras
from pandas.io.parsers import TextFileReader

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model: keras.Model, image_path: str, csv_data: TextFileReader, save=False, ):
        self.image_path = image_path
        self.model = model
        self.csv_data = csv_data
        self.save = save
        super(DisplayCallback, self).__init__()
        
    def on_epoch_end(self, epoch,  logs=None):
        # if epoch % 5 != 0:
        #  return
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))
        clear_output(wait=True)
        input_image = keras.utils.img_to_array(keras.utils.load_img(self.image_path, target_size=(128, 128))) / 255.0
        predicted_mask = self.model.predict(np.expand_dims(input_image, axis=0))
        mask_info = self.csv_data[self.csv_data['ImageId'] == '00a52cd2a.jpg']['EncodedPixels'].values
        combined_mask = ' '.join(mask_info)
        expected_mask = rle_to_mask(combined_mask)
        
        predicted_mask_uint8 = (predicted_mask[0] * 255).astype(np.uint8)

        if self.save:
            cv2.imwrite("predicted_mask_" + str(epoch) + ".png", predicted_mask_uint8)
        
        #mask_for_show = self.__create_mask(predicted_mask)
        #self.__display([input_image, expected_mask, mask_for_show])
        
    def __display(self, display_list):
        plt.figure(figsize=(15, 15))

        title = ["Input Image", "True Mask", "Predicted Mask"]

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(keras.utils.array_to_img(display_list[i]))
            plt.axis("off")
        plt.show(block=True)
        
    def __create_mask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]
