
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Dense



#  Step 1

    # defining input image shape 
    
image_shape = (100, 100, 3)

    # establishing train/validation data directories

data_folder = "Data"

train_dir = os.path.join(data_folder, "Train")
validation_dir = os.path.join(data_folder, "Validation")

    # data augmentation

train_data_gen = ImageDataGenerator(
    shear_range = 0.2,        
    zoom_range = 0.2,         
    horizontal_flip = True    
)

validation_data_gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,        
    zoom_range = 0.2,
)

    # creating train/validation generators

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size = (100, 100),
    batch_size = 32,
    label_mode = 'categorical'
)

validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size = (100, 100),
    batch_size = 32,
    label_mode = 'categorical' 
)

AUTOTUNE = tf.data.AUTOTUNE     # performance optimization
train_generator = train_generator.prefetch(buffer_size = AUTOTUNE)
validation_generator = validation_generator.prefetch(buffer_size = AUTOTUNE)



#  Step 2
    
    # building neural network
    
