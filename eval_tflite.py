from PIL import Image
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("text_xray3.jpeg").convert("RGB").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0

# Add batch dimension → (1, 224, 224, 3)
input_data = np.expand_dims(img_array, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Print prediction
print("Raw output:", output)
print("Prediction:", "Pneumonia" if output[0][0] > 0.5 else "Normal")
