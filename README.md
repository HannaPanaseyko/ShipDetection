This is a Python project for solving kaggle task Airbus Ship Detection. I used Machine Learning approach with Tensorflow and Keras.
For decreasing time for training images are doswscaled to 128x128.

Key features:
- uses all images, no additional classification of if there is a ship on image
- uses U-Net architecture

Create virtual environment

Install all requirements
pip install -r /path/to/requirements.txt

pip install -q git+https://github.com/tensorflow/examples.git

Download dataset images and csv

In file Training.py set 
IMAGES_PATH -- to path of the folder with images
CSV_PATH -- to csv file with masks
REFERENCE_IMAGE_PATH -- path to image for displaying during training, if you want to skip displaing part remove DisplayCallback from callbacks in model.fit
adjust other parameters as you wish.

For testing use file Testing.py or Testing.ipynb

Unfortunately there some known issues:
 displaying masks is borken, see saved files instead, it's recomended to remove DisplayCallback before starting training, it stops after each epoch

