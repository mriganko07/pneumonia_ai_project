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
        
        # KEY CHANGE: Convert the image to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

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
# Replace these paths with the actual paths to your images
sample_chest_image_path = "test_xray3.jpeg"
sample_other_image_path = "brain.jpeg"

# Test with your own image paths here
# my_chest_xray_path = "/path/to/your/custom_chest_xray.png"
# predict_image(my_chest_xray_path)

if os.path.exists(sample_chest_image_path):
    predict_image(sample_chest_image_path)
else:
    print("\nNote: Sample chest image not found. Please provide your own image paths.")

if os.path.exists(sample_other_image_path):
    predict_image(sample_other_image_path)
else:
    print("\nNote: Sample 'Other' image not found. Please provide your own image paths.")