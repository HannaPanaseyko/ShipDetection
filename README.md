Create virtual environment

Install all requirements
pip install -r /path/to/requirements.txt

Download dataset images and csv

In file Training.py set 
IMAGES_PATH -- to path of the folder with images
CSV_PATH -- to csv file with masks
REFERENCE_IMAGE_PATH -- path to image for displaying during training, if you want to skip displaing part remove DisplayCallback from callbacks in model.fit

For testing use file Testing.py

Unfortunately 