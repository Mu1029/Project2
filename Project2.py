
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


#  Step 1

    # defining input image shape 
    
input_shape = (100, 100, 3)

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

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size = (100, 100),
    batch_size = 32,
    label_mode = 'categorical'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size = (100, 100),
    batch_size = 32,
    label_mode = 'categorical' 
)

AUTOTUNE = tf.data.AUTOTUNE     # performance optimization
train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size = AUTOTUNE)


#  Step 2

np.random.seed(42)
tf.random.set_seed(42)

    # defining model

model = models.Sequential()

    # convolutional base

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (100, 100, 3)))
# model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))

    # flatten layer
    
model.add(layers.Flatten())

    # dense layers
    
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.55))
model.add(layers.Dense(4, activation='softmax'))


    # displaying model summary
    
#model.summary()

    # compiling model
    
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#   Step 3

    # fiting the model

history = model.fit(train_dataset, epochs=50, validation_data = validation_dataset)
    

#   Step 4

    # model accuracy and loss plots
    
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

    # test dataset accuracy and loss

test_loss, test_acc = model.evaluate(validation_dataset, verbose = 2)

print(f"\nTest Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

    # saving model
    
model.save('P2_model.h5')