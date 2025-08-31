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
# 1. Download all datasets
# ================================
paths = {
    "mooney": kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"),
    "pcb": kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images"),
    "jtiptj": kagglehub.dataset_download("jtiptj/chest-xray-pneumoniacovid19tuberculosis"),
    "prashant": kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")
}
print("✅ Datasets downloaded!")

# ================================
# 2. Create merged dataset folders
# ================================
base_dir = "merged_chest_xray"
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

def copy_images(src_dir, dst_dir, mapping):
    """Copy images into merged dataset with NORMAL/PNEUMONIA mapping"""
    for cls, target_cls in mapping.items():
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_dir, target_cls)
        if os.path.exists(src):
            for f in os.listdir(src):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))

# Mooney dataset (NORMAL / PNEUMONIA)
copy_images(os.path.join(paths["mooney"], "chest_xray/train"), os.path.join(base_dir, "train"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})
copy_images(os.path.join(paths["mooney"], "chest_xray/val"), os.path.join(base_dir, "val"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})
copy_images(os.path.join(paths["mooney"], "chest_xray/test"), os.path.join(base_dir, "test"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})

# PCB dataset (NORMAL / OPACITY → pneumonia-like)
copy_images(os.path.join(paths["pcb"], "train"), os.path.join(base_dir, "train"),
            {"normal":"NORMAL", "opacity":"PNEUMONIA"})
copy_images(os.path.join(paths["pcb"], "val"), os.path.join(base_dir, "val"),
            {"normal":"NORMAL", "opacity":"PNEUMONIA"})
copy_images(os.path.join(paths["pcb"], "test"), os.path.join(base_dir, "test"),
            {"normal":"NORMAL", "opacity":"PNEUMONIA"})

# JTiptj dataset (COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS → use only NORMAL, PNEUMONIA)
copy_images(os.path.join(paths["jtiptj"], "train"), os.path.join(base_dir, "train"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})
copy_images(os.path.join(paths["jtiptj"], "val"), os.path.join(base_dir, "val"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})
copy_images(os.path.join(paths["jtiptj"], "test"), os.path.join(base_dir, "test"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})

# Prashant dataset (Data/train/test with 3 classes)
copy_images(os.path.join(paths["prashant"], "Data/train"), os.path.join(base_dir, "train"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})
copy_images(os.path.join(paths["prashant"], "Data/test"), os.path.join(base_dir, "test"),
            {"NORMAL":"NORMAL", "PNEUMONIA":"PNEUMONIA"})

print("✅ All datasets merged!")

# ================================
# 3. Data Generators
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
# 4. Build Model (Stage 1: Freeze MobileNetV2)
# ================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False   # freeze all layers initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ================================
# 5. Stage 1 Training
# ================================
print("🔹 Stage 1 Training: only classifier layers")
history1 = model.fit(train_gen, validation_data=val_gen, epochs=5)

# ================================
# 6. Stage 2 Fine-tuning
# ================================
print("🔹 Stage 2 Training: unfreeze last 30 layers of MobileNetV2")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
history2 = model.fit(train_gen, validation_data=val_gen, epochs=5)

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
model.save("pneumonia_mobilenetv2_finetuning.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("pneumonia_model_finetuning.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model exported as TensorFlow Lite (.tflite)")
