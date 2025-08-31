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
from sklearn.model_selection import train_test_split
import os, shutil
import pandas as pd
import kagglehub

# -------------------------------
# 1. Download datasets
# -------------------------------
path_mooney = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
path_nih    = kagglehub.dataset_download("nih-chest-xrays/data")

print("Mooney Path:", path_mooney)
print("NIH Path:", path_nih)

# -------------------------------
# 2. Prepare merged dataset folders
# -------------------------------
base_dir = "merged_chest_xray"
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Copy Mooney dataset into merged dataset
def copy_images(src_dir, dst_dir):
    for cls in ["NORMAL", "PNEUMONIA"]:
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_dir, cls)
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), dst)

copy_images(os.path.join(path_mooney, "chest_xray/train"), os.path.join(base_dir, "train"))
copy_images(os.path.join(path_mooney, "chest_xray/val"), os.path.join(base_dir, "val"))
copy_images(os.path.join(path_mooney, "chest_xray/test"), os.path.join(base_dir, "test"))

print("✅ Mooney dataset copied!")

# -------------------------------
# 3. Process NIH dataset (Pneumonia + No Finding)
# -------------------------------
labels_csv = os.path.join(path_nih, "Data_Entry_2017.csv")
df = pd.read_csv(labels_csv)

# Keep only Pneumonia + No Finding
df = df[(df["Finding Labels"].str.contains("Pneumonia")) | (df["Finding Labels"] == "No Finding")]

# Balanced subset
df_pneumonia = df[df["Finding Labels"].str.contains("Pneumonia")].sample(5000, random_state=42)
df_normal    = df[df["Finding Labels"] == "No Finding"].sample(5000, random_state=42)
df_final = pd.concat([df_pneumonia, df_normal]).reset_index(drop=True)

# Index NIH image paths (they are inside images_XXX/images/)
nih_image_paths = {}
for folder in os.listdir(path_nih):
    if folder.startswith("images_"):
        img_dir = os.path.join(path_nih, folder, "images")
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                nih_image_paths[img_file] = os.path.join(img_dir, img_file)

print(f"✅ Indexed {len(nih_image_paths)} NIH images")

# Split NIH into train/val/test
train_df, temp_df = train_test_split(df_final, test_size=0.2, stratify=df_final["Finding Labels"], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Finding Labels"], random_state=42)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# Copy NIH images
missing = 0
for split, split_df in splits.items():
    for _, row in split_df.iterrows():
        label = "PNEUMONIA" if "Pneumonia" in row["Finding Labels"] else "NORMAL"
        img_file = row["Image Index"]
        if img_file in nih_image_paths:
            src = nih_image_paths[img_file]
            dst = os.path.join(base_dir, split, label, img_file)
            shutil.copy(src, dst)
        else:
            missing += 1

print(f"✅ NIH subset added! (Missing {missing} images)")

# -------------------------------
# 4. Data Generators
# -------------------------------
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='binary')
val_gen   = val_datagen.flow_from_directory(val_dir,   target_size=(224,224), batch_size=32, class_mode='binary')
test_gen  = test_datagen.flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='binary', shuffle=False)

# -------------------------------
# 5. Build Model
# -------------------------------
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

# -------------------------------
# 6. Train
# -------------------------------
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# -------------------------------
# 7. Evaluate
# -------------------------------
y_pred = model.predict(test_gen)
y_pred_classes = (y_pred > 0.5).astype("int32")

print(classification_report(test_gen.classes, y_pred_classes, target_names=["Normal","Pneumonia"]))

cm = confusion_matrix(test_gen.classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Pneumonia"], yticklabels=["Normal","Pneumonia"])
plt.show()

# -------------------------------
# 8. Save Model
# -------------------------------
model.save("pneumonia_mobilenetv2_merged.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("pneumonia_model_merged.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model exported as pneumonia_model_merged.tflite")
