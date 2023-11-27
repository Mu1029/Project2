
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


    # Step 5
    
    # loading trained model
    
model = load_model('P2_model.keras')

    # defining test data directory

data_folder = "Data"

test_dir = os.path.join(data_folder, "Test")
    
    # list of classes
    
class_names = ['Small', 'Medium', 'Large', 'None']

    # function to preprocess and predict on 1 image
    
def predict_single_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    plt.imshow(img)
    plt.title(f"Predicted Classification: {predicted_class}\n\nProbability: {100*predictions[0][predicted_class_index]:.1f}%")
    plt.show()

image_path_to_test1 = r"C:\Users\Muayad\Documents\GitHub\Project2\Data\Test\Medium\Crack__20180419_06_19_09,915.bmp"
image_path_to_test2 = r"C:\Users\Muayad\Documents\GitHub\Project2\Data\Test\Large\Crack__20180419_13_29_14,846.bmp"

predicted_class1 = predict_single_image(image_path_to_test1)
predicted_class2 = predict_single_image(image_path_to_test2)

print(f"Predicted Class for Test Image 1: {predicted_class1}")

print(f"Predicted Class for Test Image 2: {predicted_class2}")