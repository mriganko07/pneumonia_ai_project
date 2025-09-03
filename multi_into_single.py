# train_multilabel_chestxray.py
# ------------------------------------------------------------
# Multi-label Chest X-ray model that:
# - Matches Model B's strong Pneumonia accuracy (via big merged datasets)
# - Adds OTHER disease detection like Model A (via NIH labels)
# ------------------------------------------------------------

import os
import re
import gc
import math
import json
import random
import shutil
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# If not installed: pip install kagglehub
import kagglehub

# ================================
# 0. Repro & Config
# ================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 24          # keep friendly for low-end GPUs/CPUs
BASE_OUT = Path("multilabel_chestxray")
BASE_OUT.mkdir(exist_ok=True, parents=True)

# Labels we will support (extendable)
TARGET_LABELS = [
    "NORMAL",
    "PNEUMONIA",
    "COVID19",
    "TUBERCULOSIS",
    "OPACITY",
    "EFFUSION",
    "INFILTRATION",
    "ATELECTASIS",
    "CARDIOMEGALY",
    "NODULE",
    "MASS",
    "PLEURAL_THICKENING",
    "EMPHYSEMA",
    "PNEUMOTHORAX",
    "FIBROSIS",
    "HERNIA",
]
LABEL2IDX = {l:i for i,l in enumerate(TARGET_LABELS)}

# Optional: give extra emphasis to Pneumonia (helps match Model B strength)
PNEUMONIA_POS_WEIGHT = 2.0   # 1.0 disables extra weighting

# ================================
# 1. Download datasets
# ================================
print("📥 Downloading datasets via kagglehub (only first run will download)...")
paths = {
    "mooney": kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"),
    "pcb": kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images"),
    "jtiptj": kagglehub.dataset_download("jtiptj/chest-xray-pneumoniacovid19tuberculosis"),
    "prashant": kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia"),
    # NIH sample (swap to "nih-chest-xrays/data" for the full dataset if you have space)
    "nih_sample": kagglehub.dataset_download("nih-chest-xrays/sample"),
}
print("✅ Datasets ready.")

# ================================
# 2. Build a unified dataframe (multi-label)
#    We won't physically copy images; we point to original paths.
# ================================
records = []

def add_record(filepath:str, labels:list):
    labels_set = set(labels)
    # Build one-hot vector
    onehot = [1 if lab in labels_set else 0 for lab in TARGET_LABELS]
    records.append({
        "filepath": str(filepath),
        **{lab: onehot[i] for lab,i in LABEL2IDX.items()}
    })

def add_from_folder_binary(base_dir, split_subdirs, mapping):
    """
    mapping: dict like {'NORMAL': 'NORMAL', 'PNEUMONIA':'PNEUMONIA'}
    split_subdirs: ['train','val','test'] or dataset-specific
    For each split, look for subfolders of classes listed in mapping keys.
    """
    for split in split_subdirs:
        split_dir = Path(base_dir) / split
        if not split_dir.exists():
            continue
        for src_cls, target_cls in mapping.items():
            src_dir = split_dir / src_cls
            if not src_dir.exists(): 
                # robust to case differences
                alt = split_dir / src_cls.lower()
                if alt.exists(): src_dir = alt
                else: continue
            for f in src_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                    labs = ["NORMAL"] if target_cls == "NORMAL" else [target_cls]
                    add_record(f, labs)

# --- Mooney: chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
mooney_root = Path(paths["mooney"]) / "chest_xray"
add_from_folder_binary(mooney_root, ["train","val","test"], {"NORMAL":"NORMAL","PNEUMONIA":"PNEUMONIA"})

# --- PCB: {train,val,test}/{normal,opacity}
pcb_root = Path(paths["pcb"])
add_from_folder_binary(pcb_root, ["train","val","test"], {"normal":"NORMAL","opacity":"OPACITY"})

# --- JTiptj: {train,val,test}/{COVID19,NORMAL,PNEUMONIA,TUBERCULOSIS}
jt_root = Path(paths["jtiptj"])
# Use all four classes (COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS)
for split in ["train","val","test"]:
    for cls in ["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]:
        src_dir = jt_root / split / cls
        if src_dir.exists():
            for f in src_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                    add_record(f, [cls])

# --- Prashant: Data/train and Data/test with {COVID19,NORMAL,PNEUMONIA}
pr_root = Path(paths["prashant"]) / "Data"
for split in ["train","test"]:
    for cls in ["COVID19","NORMAL","PNEUMONIA"]:
        src_dir = pr_root / split / cls
        if src_dir.exists():
            for f in src_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                    add_record(f, [cls])

# --- NIH sample: sample/sample_labels.csv + sample/images
nih_root = Path(paths["nih_sample"])
nih_csv = nih_root / "sample_labels.csv"
nih_img_dir = nih_root / "sample" / "images"

# Map NIH findings (underscored names) to TARGET_LABELS
NIH_MAP = {
    "Atelectasis": "ATELECTASIS",
    "Cardiomegaly": "CARDIOMEGALY",
    "Effusion": "EFFUSION",
    "Infiltration": "INFILTRATION",
    "Mass": "MASS",
    "Nodule": "NODULE",
    "Pneumonia": "PNEUMONIA",
    "Pneumothorax": "PNEUMOTHORAX",
    "Consolidation": "OPACITY",          # map consolidation-like to OPACITY bucket
    "Edema": "OPACITY",
    "Emphysema": "EMPHYSEMA",
    "Fibrosis": "FIBROSIS",
    "Pleural_Thickening": "PLEURAL_THICKENING",
    "Hernia": "HERNIA",
}

if nih_csv.exists():
    df_nih = pd.read_csv(nih_csv)
    for _, row in df_nih.iterrows():
        img = nih_img_dir / row["Image Index"]
        if not img.exists(): 
            continue
        findings = str(row.get("Finding Labels","")).strip()
        if findings in ["", "No Finding", "null", "nan"]:
            add_record(img, ["NORMAL"])
        else:
            labs = []
            for token in [t.strip() for t in findings.split("|") if t.strip()]:
                mapped = NIH_MAP.get(token)
                if mapped is not None and mapped in TARGET_LABELS:
                    labs.append(mapped)
            if not labs:
                # if labels are outside our reduced set, skip or mark as NORMAL? We'll skip.
                continue
            add_record(img, labs)

# Build dataframe
df = pd.DataFrame.from_records(records)
print(f"🧾 Total images collected: {len(df):,}")

# Drop potential duplicates by filepath
df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
print(f"🧹 After duplicate drop: {len(df):,}")

# ================================
# 3. Train/Val/Test Split (multi-label)
#    We'll do a simple random split; optionally stratify on Pneumonia presence.
# ================================
has_pneu = df["PNEUMONIA"].values
df_train, df_tmp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=has_pneu)
has_pneu_tmp = df_tmp["PNEUMONIA"].values
df_val, df_test = train_test_split(df_tmp, test_size=0.50, random_state=SEED, stratify=has_pneu_tmp)

for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"{name:>5}: {len(part):,} images | Pneumonia %+ = {part['PNEUMONIA'].mean():.3f}")

# Save splits (optional)
df_train.to_csv(BASE_OUT/"train.csv", index=False)
df_val.to_csv(BASE_OUT/"val.csv", index=False)
df_test.to_csv(BASE_OUT/"test.csv", index=False)

# ================================
# 4. tf.data pipeline
# ================================
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Keras preprocessing augmentation (GPU/CPU friendly)
augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
], name="augmenter")

def make_ds(frame: pd.DataFrame, shuffle=False, augment=False, batch_size=BATCH_SIZE):
    paths = frame["filepath"].values
    labels = frame[TARGET_LABELS].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(frame), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(df_train, shuffle=True, augment=True)
val_ds   = make_ds(df_val, shuffle=False, augment=False)
test_ds  = make_ds(df_test, shuffle=False, augment=False)

# ================================
# 5. Build Model (MobileNetV2 → GAP → Dropout → Dense(sigmoid))
# ================================
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3), alpha=0.5)
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(TARGET_LABELS), activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=outputs)

# Stage 1: freeze backbone
for l in base.layers:
    l.trainable = False

METRICS = [
    tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=len(TARGET_LABELS), name="auc_macro"),
    tf.keras.metrics.AUC(curve="PR",  multi_label=True, num_labels=len(TARGET_LABELS), name="auprc_macro"),
    tf.keras.metrics.Precision(name="precision", thresholds=0.5),
    tf.keras.metrics.Recall(name="recall", thresholds=0.5),
]

model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=METRICS)
model.summary()

# ================================
# 6. Optional: per-sample weights to boost Pneumonia positives
# ================================
def make_sample_weights(frame: pd.DataFrame, pneumonia_boost=PNEUMONIA_POS_WEIGHT):
    """
    Simple scheme: if sample is pneumonia-positive, weight it higher.
    Otherwise weight = 1.0
    """
    if pneumonia_boost <= 1.0:
        return np.ones(len(frame), dtype="float32")
    w = np.ones(len(frame), dtype="float32")
    pneu_pos = frame["PNEUMONIA"].values.astype(bool)
    w[pneu_pos] = pneumonia_boost
    return w

train_weights = make_sample_weights(df_train, PNEUMONIA_POS_WEIGHT)
val_weights   = np.ones(len(df_val), dtype="float32")

# Convert weights into tf.data by zipping
w_train_ds = tf.data.Dataset.from_tensor_slices(train_weights).batch(BATCH_SIZE)
w_val_ds   = tf.data.Dataset.from_tensor_slices(val_weights).batch(BATCH_SIZE)

train_ds_w = tf.data.Dataset.zip((train_ds, w_train_ds))
val_ds_w   = tf.data.Dataset.zip((val_ds,   w_val_ds))

# Keras expects ((x,y), sample_weight)
train_ds_w = train_ds_w.map(lambda xy, w: (xy[0], xy[1], w))
val_ds_w   = val_ds_w.map(lambda xy, w: (xy[0], xy[1], w))

# ================================
# 7. Train: Stage 1 (frozen), Stage 2 (fine-tune)
# ================================
ckpt_path = str(BASE_OUT / "best_multilabel_mnv2.h5")

callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_auc_macro", mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_auc_macro", mode="max", patience=2, factor=0.5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_auc_macro", mode="max", patience=5, restore_best_weights=True, verbose=1),
]

print("🔒 Stage 1: training classifier head (backbone frozen)")
history1 = model.fit(
    train_ds_w,
    validation_data=val_ds_w,
    epochs=6,
    verbose=1,
    callbacks=callbacks
)

print("🔓 Stage 2: fine-tuning last 30 layers")
for l in base.layers[-30:]:
    l.trainable = True

model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=METRICS)

history2 = model.fit(
    train_ds_w,
    validation_data=val_ds_w,
    epochs=10,
    verbose=1,
    callbacks=callbacks
)

# Load best checkpoint
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)

# ================================
# 8. Evaluate (per-class report at threshold 0.5)
# ================================
print("📊 Evaluating on test set...")
y_true = df_test[TARGET_LABELS].values.astype(int)

# Collect predictions
y_prob = []
for x, y in test_ds:
    y_prob.append(model.predict(x, verbose=0))
y_prob = np.vstack(y_prob)

# Threshold
thr = 0.5
y_pred = (y_prob >= thr).astype(int)

print("\n=== Classification report (threshold = 0.5) ===")
print(classification_report(y_true, y_pred, target_names=TARGET_LABELS, zero_division=0))

# Save per-class AUROC/AUPRC quickly
def per_class_auc(y_true, y_prob):
    # Fallback simple per-class AUC using tf.metrics for consistency
    auroc = []
    auprc = []
    for i, lab in enumerate(TARGET_LABELS):
        # skip if all zeros or ones in ground truth
        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == y_true.shape[0]:
            auroc.append(np.nan); auprc.append(np.nan); continue
        m1 = tf.keras.metrics.AUC(curve="ROC")
        m2 = tf.keras.metrics.AUC(curve="PR")
        m1.update_state(y_true[:, i], y_prob[:, i])
        m2.update_state(y_true[:, i], y_prob[:, i])
        auroc.append(float(m1.result().numpy()))
        auprc.append(float(m2.result().numpy()))
    return auroc, auprc

auroc, auprc = per_class_auc(y_true, y_prob)
per_metrics = pd.DataFrame({"label": TARGET_LABELS, "AUROC": auroc, "AUPRC": auprc})
per_metrics.to_csv(BASE_OUT/"per_class_metrics.csv", index=False)
print("\nSaved per-class AUROC/AUPRC to:", BASE_OUT/"per_class_metrics.csv")

# ================================
# 9. Save Keras + TFLite (quantized)
# ================================
keras_path = str(BASE_OUT / "multilabel_mnv2.h5")
print("💾 Saving Keras model to", keras_path)
model.save(keras_path)

print("🔧 Converting to TFLite (float16 quantization)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
try:
    converter.target_spec.supported_types = [tf.float16]
except Exception:
    pass
tflite_model = converter.convert()
tflite_path = str(BASE_OUT / "multilabel_mnv2_fp16.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("🎉 Exported:", tflite_path)

# ================================
# 10. Export label map & threshold (for inference)
# ================================
with open(BASE_OUT/"labels.json", "w") as f:
    json.dump({"labels": TARGET_LABELS, "threshold": thr}, f, indent=2)
print("🗂  Saved labels.json")

print("\n✅ Done. You now have a multi-label model with strong Pneumonia performance and other-disease detection.")
