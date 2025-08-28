import os
import shutil
import random
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------
# 1. Download datasets
# -------------------------
pneumonia_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
animals_path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Pneumonia dataset:", pneumonia_path)
print("Animals dataset:", animals_path)

# -------------------------
# 2. Create new dataset structure
# -------------------------
base_dir = "preclassifier_data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

for d in [train_dir, val_dir]:
    os.makedirs(os.path.join(d, "xray"), exist_ok=True)
    os.makedirs(os.path.join(d, "non_xray"), exist_ok=True)

# Pneumonia (X-ray images)
xray_train_src = os.path.join(pneumonia_path, "chest_xray/train")
xray_val_src   = os.path.join(pneumonia_path, "chest_xray/val")

# Animals (Non-Xray images)
animals_src = os.path.join(animals_path, "raw-img")

def copy_images(src_dirs, dst_dir, label, limit=None):
    """Copy images from multiple class folders into one label folder"""
    all_imgs = []
    for folder in src_dirs:
        folder_path = os.path.join(src_dirs_base, folder)
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_imgs.append(os.path.join(root, f))
    if limit:
        all_imgs = random.sample(all_imgs, limit)
    for i, img_path in enumerate(all_imgs):
        shutil.copy(img_path, os.path.join(dst_dir, label, f"{label}_{i}.jpg"))

# -------------------------
# 3. Copy data
# -------------------------
# X-rays → "xray"
for cls in ["NORMAL", "PNEUMONIA"]:
    src_cls = os.path.join(xray_train_src, cls)
    for f in os.listdir(src_cls):
        shutil.copy(os.path.join(src_cls, f), os.path.join(train_dir, "xray", f))

for cls in ["NORMAL", "PNEUMONIA"]:
    src_cls = os.path.join(xray_val_src, cls)
    for f in os.listdir(src_cls):
        shutil.copy(os.path.join(src_cls, f), os.path.join(val_dir, "xray", f))

# Non-Xray → "non_xray"
for animal_class in os.listdir(animals_src):
    src_cls = os.path.join(animals_src, animal_class)
    if os.path.isdir(src_cls):
        files = os.listdir(src_cls)
        random.shuffle(files)
        split = int(0.8 * len(files))  # 80% train, 20% val
        for f in files[:split]:
            shutil.copy(os.path.join(src_cls, f), os.path.join(train_dir, "non_xray", f"{animal_class}_{f}"))
        for f in files[split:]:
            shutil.copy(os.path.join(src_cls, f), os.path.join(val_dir, "non_xray", f"{animal_class}_{f}"))

print("✅ Dataset built at:", base_dir)

# -------------------------
# 4. Train Pre-Classifier
# -------------------------
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode="binary")
val_gen   = datagen.flow_from_directory(val_dir,   target_size=(224,224), batch_size=32, class_mode="binary")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# -------------------------
# 5. Save model
# -------------------------
model.save("xray_preclassifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("xray_preclassifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Pre-classifier exported as .h5 and .tflite")
