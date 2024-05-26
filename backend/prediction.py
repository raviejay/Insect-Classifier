import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = 'resources/dataset/trained_model.h5'
class_names_path = 'resources/dataset/Classname.txt'

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Load class names
try:
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()
    print("Class names loaded successfully.")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# Function to preprocess the image and make a prediction
def predict_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class
