import glob
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor
import os
from skimage import io

dataset_path = "train_v2"
imgs_names = os.listdir(dataset_path)

def process_image(imgname):
    try:
        img = cv2.imread(imgname)
        _ = io.imread(imgname)
        if _ is None:
            print(imgname)

        if img is None:
            print(imgname)
        # if size iz not 768x768 print name
        if img.shape != (768, 768, 3):
            print(imgname)
        
        # if image is empty print name
        if img.size == 0:
            print(imgname)
        
        # if image is not RGB print name
        if img.ndim != 3:
            print(imgname)

    except (IOError, cv2.error, Exception) as e:
        print(imgname)
        print(e)

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(process_image, imgs_names)