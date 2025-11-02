import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ==========================================
# 1. Load the TFLite model
# ==========================================
try:
    interpreter = tf.lite.Interpreter(model_path="/home/mriganka/Downloads/xray_preclassifier_new.tflite")
    interpreter.allocate_tensors()
except ValueError:
    print("Error: Could not load the TFLite model. Please ensure the file exists at /content/xray_preclassifier.tflite.")
    exit()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
img_size = (input_shape[1], input_shape[2])

# ==========================================
# 2. Function to predict a single image
# ==========================================
def predict_image(image_path):
    """
    Predicts whether an X-ray image is a Chest or Other.
    """
    try:
        # Load and resize the image
        img = Image.open(image_path)
        img = img.resize(img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # Preprocess the image to match the model's input
        # The model includes a Rescaling layer (1./255), so no manual division is needed.

        # Set the tensor and invoke inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get the prediction result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0][0]

        # Interpret the result
        class_label = "Chest" if prediction < 0.5 else "Other"
        confidence = prediction if class_label == "Other" else 1 - prediction

        print(f"\nPrediction for {image_path}:")
        print(f"Class: {class_label}")
        print(f"Confidence: {confidence:.2f}")
        return class_label, confidence

    except FileNotFoundError:
        print(f"Error: The image file was not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# ==========================================
# 3. Example usage
# ==========================================
# Placeholder paths for demonstration. Replace with your actual image paths.
chest_xray_path = "test_xray.jpeg" 
other_xray_path = "hand1.jpeg" 

# Use a sample chest image from the dataset for demonstration purposes.
# This assumes the code from the original prompt has been run and the data is available.
sample_chest_image_path = "test_xray3.jpeg"
if not os.path.exists(sample_chest_image_path):
    print("\nNote: Sample image not found. Please provide your own image paths.")
else:
    predict_image(sample_chest_image_path)

# Use a sample 'Other' image for demonstration purposes.
sample_other_image_path = "hand1.jpeg"
if not os.path.exists(sample_other_image_path):
    print("\nNote: Sample image not found. Please provide your own image paths.")
else:
    predict_image(sample_other_image_path)