import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os, shutil
import kagglehub

# ================================
# 1. Download datasets
# ================================
datasets = {
    "mooney": "paultimothymooney/chest-xray-pneumonia",
    "pcb": "pcbreviglieri/pneumonia-xray-images",
    "jtip": "jtiptj/chest-xray-pneumoniacovid19tuberculosis",
    "prashant": "prashant268/chest-xray-covid19-pneumonia"
}

downloaded_paths = {}
for name, kaggle_id in datasets.items():
    path = kagglehub.dataset_download(kaggle_id)
    downloaded_paths[name] = path
    print(f"✅ Downloaded {name}: {path}")

# ================================
# 2. Prepare merged dataset folder
# ================================
base_dir = "merged_chest_xray"
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

def copy_images(src_dir, dst_dir, valid_classes):
    """
    Copy images from src_dir into dst_dir, keeping only valid_classes.
    """
    if not os.path.exists(src_dir):
        return
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        cls_upper = cls.upper()
        if cls_upper == "OPACITY":  # special case in pcb dataset
            cls_upper = "PNEUMONIA"
        if cls_upper in valid_classes:
            dst_class = os.path.join(dst_dir, cls_upper)
            os.makedirs(dst_class, exist_ok=True)
            for file in os.listdir(cls_path):
                shutil.copy(os.path.join(cls_path, file), dst_class)

# ================================
# 3. Merge datasets
# ================================

# Mooney dataset
copy_images(os.path.join(downloaded_paths["mooney"], "chest_xray/train"), os.path.join(base_dir, "train"), ["NORMAL","PNEUMONIA"])
copy_images(os.path.join(downloaded_paths["mooney"], "chest_xray/val"), os.path.join(base_dir, "val"), ["NORMAL","PNEUMONIA"])
copy_images(os.path.join(downloaded_paths["mooney"], "chest_xray/test"), os.path.join(base_dir, "test"), ["NORMAL","PNEUMONIA"])

# PCB dataset
copy_images(os.path.join(downloaded_paths["pcb"], "train"), os.path.join(base_dir, "train"), ["NORMAL","OPACITY"])
copy_images(os.path.join(downloaded_paths["pcb"], "val"), os.path.join(base_dir, "val"), ["NORMAL","OPACITY"])
copy_images(os.path.join(downloaded_paths["pcb"], "test"), os.path.join(base_dir, "test"), ["NORMAL","OPACITY"])

# JTipt dataset (ignore COVID19 + TUBERCULOSIS)
copy_images(os.path.join(downloaded_paths["jtip"], "train"), os.path.join(base_dir, "train"), ["NORMAL","PNEUMONIA"])
copy_images(os.path.join(downloaded_paths["jtip"], "val"), os.path.join(base_dir, "val"), ["NORMAL","PNEUMONIA"])
copy_images(os.path.join(downloaded_paths["jtip"], "test"), os.path.join(base_dir, "test"), ["NORMAL","PNEUMONIA"])

# Prashant dataset
copy_images(os.path.join(downloaded_paths["prashant"], "Data/train"), os.path.join(base_dir, "train"), ["NORMAL","PNEUMONIA"])
copy_images(os.path.join(downloaded_paths["prashant"], "Data/test"), os.path.join(base_dir, "test"), ["NORMAL","PNEUMONIA"])

print("✅ All datasets merged into:", base_dir)

# ================================
# 4. Data Generators
# ================================
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(os.path.join(base_dir, "train"), target_size=(224,224), batch_size=32, class_mode='binary')
val_gen   = val_datagen.flow_from_directory(os.path.join(base_dir, "val"),   target_size=(224,224), batch_size=32, class_mode='binary')
test_gen  = test_datagen.flow_from_directory(os.path.join(base_dir, "test"), target_size=(224,224), batch_size=32, class_mode='binary', shuffle=False)

# ================================
# 5. Build Model
# ================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ================================
# 6. Train
# ================================
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# ================================
# 7. Evaluate
# ================================
y_pred = model.predict(test_gen)
y_pred_classes = (y_pred > 0.5).astype("int32")

print(classification_report(test_gen.classes, y_pred_classes, target_names=["Normal","Pneumonia"]))

cm = confusion_matrix(test_gen.classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Pneumonia"], yticklabels=["Normal","Pneumonia"])
plt.show()

# ================================
# 8. Save Model
# ================================
model.save("pneumonia_mobilenetv2_merged.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("pneumonia_model_merged.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model exported as .h5 and .tflite")
