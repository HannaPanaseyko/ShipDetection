Create virtual environment

Install all requirements
pip install -r /path/to/requirements.txt

Download dataset images and csv

In file Training.py set 
IMAGES_PATH -- to path of the folder with images
CSV_PATH -- to csv file with masks
REFERENCE_IMAGE_PATH -- path to image for displaying during training, if you want to skip displaing part remove DisplayCallback from callbacks in model.fit
adjust other parameters as you wish.

For testing use file Testing.py

Unfortunately there some known issues:
 displaying masks is borken, see saved files instead, it's recomended to remove DisplayCallback before starting training, it stops after each epoch
 Testing is a bit broken, but you can see result on some image during training, i used image from dataset, not real-world scenario but it was good for dev purposes.