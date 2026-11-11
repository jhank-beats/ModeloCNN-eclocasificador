#!/usr/bin/env python3
"""
Script de Reentrenamiento y Corrección de Generalización In-The-Wild (EfficientNetV2-B0).
Resuelve el sobreajuste al fondo de estudio RealWaste y mitiga la confusión inter-clase en:
  1. Cardboard vs. Madera_Procesada
  2. Food_Organics vs. Medicamentos / Tetra_Pack
  3. Glass vs. Aceites_Quimicos / Bombillas
  4. Metal vs. Pilas_Baterias / Bombillas / RAEE_Electronicos / Madera_Procesada
  5. Miscellaneous_Trash
  6. Paper
  7. Plastic vs. Bombillas / Aceites_Quimicos / Tetra_Pack
  8. Textile_Trash vs. Residuos_Sanitarios / Neumaticos_Gomas / RAEE_Electronicos
  9. Vegetation
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_completo")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
FLUTTER_ASSETS_DIR = os.path.join(BASE_DIR, "momento tres", "semana 13", "flutter_app_residuos", "assets")

IMG_SIZE = (180, 180)
BATCH_SIZE = 32

class_names = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
])
NUM_CLASSES = len(class_names)
print(f"📦 Clases detectadas ({NUM_CLASSES}): {class_names}")

# 1. Recolección del Dataset Enriquecido (RealWaste + Debiased + Wild Internet)
registros = []
for idx, cname in enumerate(class_names):
    cdir = os.path.join(DATASET_DIR, cname)
    for fname in os.listdir(cdir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            fpath = os.path.join(cdir, fname)
            es_wild = fname.startswith("wild_")
            es_debiased = fname.startswith("aug_debiased_")
            # Clave de estratificación combinada (clase + origen)
            origen = "wild" if es_wild else ("debiased" if es_debiased else "base")
            registros.append({
                "ruta": fpath,
                "clase": cname,
                "label": idx,
                "origen": origen,
                "strat_key": f"{idx}_{origen}"
            })

df = pd.DataFrame(registros)
print(f"📊 Total de imágenes válidas en dataset enriquecido: {len(df)}")
print(df.groupby(["clase", "origen"]).size().unstack(fill_value=0))

# Partición estratificada 82% Train / 9% Val / 9% Test
df_train, df_temp = train_test_split(
    df, test_size=0.18, random_state=SEED, stratify=df["strat_key"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["strat_key"]
)

print(f" -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# 2. Pesos de clase con refuerzo específico para las 9 clases problemáticas en Wild
y_train = df_train["label"].values
base_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(base_weights)}

# Multiplicadores de des-sesgo de dominio para erradicar falsos negativos en las 9 clases
BOOST_FACTORS = {
    "Cardboard": 1.35,
    "Food_Organics": 1.40,
    "Glass": 1.45,
    "Metal": 1.35,
    "Miscellaneous_Trash": 1.35,
    "Paper": 1.45,
    "Plastic": 1.35,
    "Textile_Trash": 1.50,
    "Vegetation": 1.35,
    # Moderación de clases sobre-predichas (falsos positivos frecuentes)
    "Aceites_Quimicos": 0.82,
    "Bombillas": 0.82,
    "Madera_Procesada": 0.85,
    "Medicamentos": 0.85,
    "Neumaticos_Gomas": 0.82,
    "Pilas_Baterias": 0.88,
    "RAEE_Electronicos": 0.85,
    "Residuos_Sanitarios": 0.80,
    "Tetra_Pack": 0.78,
}

for idx, cname in enumerate(class_names):
    factor = BOOST_FACTORS.get(cname, 1.0)
    class_weights[idx] = round(class_weights[idx] * factor, 4)

# Vector de pesos por clase en TensorFlow para la función de pérdida
cw_vector = tf.constant([class_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

# 3. Pipeline tf.data con Data Augmentation de Invarianza de Fondo y Textura
def decodificar_imagen(ruta, y):
    img_bytes = tf.io.read_file(ruta)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.float32)
    img.set_shape((*IMG_SIZE, 3))
    return img, y

aug_pipeline = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.18, fill_mode="reflect"),
    layers.RandomZoom(height_factor=(-0.22, 0.10), fill_mode="reflect"),
    layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
    layers.RandomBrightness(0.20, value_range=(0, 255)),
    layers.RandomContrast(0.22),
], name="wild_domain_augmentation")

def build_dataset(df_sub, training=False):
    rutas = df_sub["ruta"].values
    # Duplicar peso muestral de imágenes "wild" en entrenamiento para acelerar convergencia en campo
    if training:
        wild_mask = (df_sub["origen"] == "wild").values
        rutas_wild = rutas[wild_mask]
        labels_wild = df_sub["label"].values[wild_mask]
        rutas = np.concatenate([rutas, rutas_wild])
        labels_all = np.concatenate([df_sub["label"].values, labels_wild])
    else:
        labels_all = df_sub["label"].values

    y_cat = keras.utils.to_categorical(labels_all, NUM_CLASSES).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((rutas, y_cat))
    if training:
        ds = ds.shuffle(len(rutas), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decodificar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    if training:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(aug_pipeline(x, training=True), 0.0, 255.0), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(df_train, training=True)
val_ds = build_dataset(df_val, training=False)
test_ds = build_dataset(df_test, training=False)

# Subconjunto de prueba exclusivamente de imágenes Wild (In-The-Wild)
df_test_wild = df_test[df_test["origen"] == "wild"].reset_index(drop=True)
test_wild_ds = build_dataset(df_test_wild, training=False)

# 4. Función de Pérdida Focal Sensible a Confusión Inter-Clase
def confusion_aware_focal_loss(gamma=2.0, label_smoothing=0.04):
    def loss_fn(y_true, y_pred):
        y_true_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / float(NUM_CLASSES))
        y_pred_clip = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_smooth * tf.math.log(y_pred_clip)
        focal_weight = tf.pow(1.0 - y_pred_clip, gamma)
        weighted_focal = focal_weight * cross_entropy * cw_vector[None, :]
        return tf.reduce_sum(weighted_focal, axis=-1)
    return loss_fn

# 5. Cargar y Re-entrenar Modelo Principal (Focal Loss) y Modelo Base (Ensamble)
path_focal = os.path.join(MODELOS_DIR, "modelo_clasificador_19_clases_focal_loss.keras")
path_base = os.path.join(MODELOS_DIR, "modelo_clasificador_19_clases_efficientnetv2b0.keras")

print("\n🧠 Cargando modelo Focal Loss para Fine-Tuning de Generalización In-The-Wild...")
model_focal = keras.models.load_model(path_focal, compile=False)

# Descongelar capas superiores para adaptar filtros de textura y transparencia
for layer in model_focal.layers:
    if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
        layer.trainable = True
        for sublayer in layer.layers[:-65]:
            sublayer.trainable = False
        for sublayer in layer.layers[-65:]:
            if not isinstance(sublayer, layers.BatchNormalization):
                sublayer.trainable = True
    elif not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

steps_per_epoch = len(train_ds)
EPOCHS_FOCAL = 8

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1.8e-4,
    decay_steps=EPOCHS_FOCAL * steps_per_epoch,
    alpha=0.05
)

model_focal.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=confusion_aware_focal_loss(gamma=2.0, label_smoothing=0.04),
    metrics=["accuracy"]
)

t0 = time.time()
history_focal = model_focal.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FOCAL,
    verbose=2
)
print(f"✅ Fine-tuning de model_focal completado en {time.time() - t0:.1f}s")
model_focal.save(path_focal)
print(f"💾 Guardado en: {path_focal}")

print("\n🧠 Ajustando modelo Base (EfficientNetV2-B0) para el Ensamble...")
model_base = keras.models.load_model(path_base, compile=False)
for layer in model_base.layers:
    if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
        layer.trainable = True
        for sublayer in layer.layers[:-50]:
            sublayer.trainable = False
        for sublayer in layer.layers[-50:]:
            if not isinstance(sublayer, layers.BatchNormalization):
                sublayer.trainable = True
    elif not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

EPOCHS_BASE = 5
lr_schedule_base = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1.2e-4,
    decay_steps=EPOCHS_BASE * steps_per_epoch,
    alpha=0.05
)
model_base.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule_base, weight_decay=1e-4),
    loss=confusion_aware_focal_loss(gamma=1.5, label_smoothing=0.05),
    metrics=["accuracy"]
)
model_base.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_BASE,
    verbose=2
)
model_base.save(path_base)
print(f"💾 Guardado en: {path_base}")

# 6. Exportar a TFLite FP16 para la App Móvil Flutter
print("\n📱 Exportando nuevo modelo optimizado a TFLite (FP16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model_focal)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()

tflite_path_main = os.path.join(MODELOS_DIR, "modelo_efficientnetv2b0_fp16.tflite")
with open(tflite_path_main, "wb") as f:
    f.write(tflite_fp16)

if os.path.exists(FLUTTER_ASSETS_DIR):
    tflite_flutter_path = os.path.join(FLUTTER_ASSETS_DIR, "modelo_clasificador_19_clases.tflite")
    with open(tflite_flutter_path, "wb") as f:
        f.write(tflite_fp16)
    print(f"✅ TFLite actualizado en Flutter assets: {tflite_flutter_path}")

# 7. Evaluación Rigurosa en Test Global y Test Wild (9 Clases Problemáticas)
print("\n📊 Evaluando Ensamble Calibrado (80% Focal + 20% Base) en Test Global y Test Wild...")
y_true_test = df_test["label"].values
p_f = model_focal.predict(test_ds, verbose=0)
p_b = model_base.predict(test_ds, verbose=0)
p_ens = 0.80 * p_f + 0.20 * p_b
y_pred_test = np.argmax(p_ens, axis=1)

acc_global = accuracy_score(y_true_test, y_pred_test)
f1_global = f1_score(y_true_test, y_pred_test, average="macro")
print(f"🌟 Exactitud Test Global (19 clases): {acc_global*100:.2f}% | Macro F1: {f1_global:.4f}")

if len(df_test_wild) > 0:
    y_true_w = df_test_wild["label"].values
    p_fw = model_focal.predict(test_wild_ds, verbose=0)
    p_bw = model_base.predict(test_wild_ds, verbose=0)
    p_ens_w = 0.80 * p_fw + 0.20 * p_bw
    y_pred_w = np.argmax(p_ens_w, axis=1)
    acc_wild = accuracy_score(y_true_w, y_pred_w)
    print(f"🔥 Exactitud Test In-The-Wild (9 clases antes fallidas): {acc_wild*100:.2f}%")
    present_labels = sorted(list(set(y_true_w)))
    present_names = [class_names[i] for i in present_labels]
    print(classification_report(y_true_w, y_pred_w, labels=present_labels, target_names=present_names, digits=4))

