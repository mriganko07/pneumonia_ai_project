import os
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

# =============================
# STEP 0: Download NIH Sample Dataset
# =============================
print("📥 Downloading NIH Chest X-ray Sample Dataset...")
path = kagglehub.dataset_download("nih-chest-xrays/sample")
print("✅ Downloaded NIH sample dataset at:", path)

CSV_PATH = os.path.join(path, "sample_labels.csv")
IMAGES_DIR = os.path.join(path, "sample", "images")
BASE_DIR = "nih_sample_binary"   # where processed dataset will go

# =============================
# STEP 1: Prepare dataset folders
# =============================
df = pd.read_csv(CSV_PATH)

# Binary labels: Pneumonia vs Other
df["Label"] = df["Finding Labels"].apply(lambda x: "PNEUMONIA" if "Pneumonia" in x else "OTHER")

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split ratios
train_split, val_split = 0.7, 0.15
n = len(df)
train_end = int(train_split * n)
val_end = int(val_split * n) + train_end

splits = {
    "train": df.iloc[:train_end],
    "val":   df.iloc[train_end:val_end],
    "test":  df.iloc[val_end:]
}

# Create folders
for split in splits:
    for cls in ["PNEUMONIA", "OTHER"]:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Copy images
for split, subset in splits.items():
    for _, row in subset.iterrows():
        src = os.path.join(IMAGES_DIR, row["Image Index"])
        dst = os.path.join(BASE_DIR, split, row["Label"], row["Image Index"])
        if os.path.exists(src):
            shutil.copy(src, dst)

print("✅ Dataset prepared at:", BASE_DIR)

# =============================
# STEP 2: Data Generators
# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16   # keep small for low-end devices

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="binary"
)
val_gen = val_test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="binary"
)
test_gen = val_test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, "test"), target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
)

# =============================
# STEP 3: Build Lightweight Model
# =============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3), alpha=0.35)  
# alpha=0.35 makes it even smaller

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

# Phase 1: Freeze base
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
print("🔒 Training with frozen base...")
history1 = model.fit(train_gen, validation_data=val_gen, epochs=3)

# Phase 2: Fine-tune last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
print("🔓 Fine-tuning last 20 layers...")
history2 = model.fit(train_gen, validation_data=val_gen, epochs=5)

# =============================
# STEP 4: Evaluate
# =============================
y_pred = model.predict(test_gen)
y_pred_classes = (y_pred > 0.5).astype("int32")

print(classification_report(test_gen.classes, y_pred_classes, target_names=["OTHER","PNEUMONIA"]))

cm = confusion_matrix(test_gen.classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["OTHER","PNEUMONIA"], yticklabels=["OTHER","PNEUMONIA"])
plt.show()

# =============================
# STEP 5: Save Optimized Model
# =============================
model.save("nih_sample_pneumonia_model.h5")

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization
tflite_model = converter.convert()

with open("nih_sample_pneumonia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("🎉 Optimized model exported as .h5 and quantized .tflite")
