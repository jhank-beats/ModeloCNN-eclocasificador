#!/usr/bin/env python3
"""
Segunda etapa de Fine-Tuning Profundo (Full Backbone Unfreeze con LR bajo) +
Exportación limpia a TFLite FP16 mediante clonación en política float32.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

registros = []
for idx, cname in enumerate(class_names):
    cdir = os.path.join(DATASET_DIR, cname)
    for fname in os.listdir(cdir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            fpath = os.path.join(cdir, fname)
            es_wild = fname.startswith("wild_")
            es_debiased = fname.startswith("aug_debiased_")
            origen = "wild" if es_wild else ("debiased" if es_debiased else "base")
            registros.append({
                "ruta": fpath,
                "clase": cname,
                "label": idx,
                "origen": origen,
                "strat_key": f"{idx}_{origen}"
            })

df = pd.DataFrame(registros)
df_train, df_temp = train_test_split(df, test_size=0.18, random_state=SEED, stratify=df["strat_key"])
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=SEED, stratify=df_temp["strat_key"])

y_train = df_train["label"].values
base_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(base_weights)}

BOOST_STAGE2 = {
    "Cardboard": 1.55,
    "Food_Organics": 1.50,
    "Glass": 1.85,
    "Metal": 1.40,
    "Miscellaneous_Trash": 1.80,
    "Paper": 1.55,
    "Plastic": 1.85,
    "Textile_Trash": 1.85,
    "Vegetation": 1.40,
    "Aceites_Quimicos": 0.78,
    "Bombillas": 0.78,
    "Madera_Procesada": 0.80,
    "Medicamentos": 0.82,
    "Neumaticos_Gomas": 0.78,
    "Pilas_Baterias": 0.85,
    "RAEE_Electronicos": 0.80,
    "Residuos_Sanitarios": 0.75,
    "Tetra_Pack": 0.75,
}
for idx, cname in enumerate(class_names):
    class_weights[idx] = round(class_weights[idx] * BOOST_STAGE2.get(cname, 1.0), 4)

cw_vector = tf.constant([class_weights[i] for i in range(NUM_CLASSES)], dtype=tf.float32)

def decodificar_imagen(ruta, y):
    img_bytes = tf.io.read_file(ruta)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.float32)
    img.set_shape((*IMG_SIZE, 3))
    return img, y

aug_pipeline = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15, fill_mode="reflect"),
    layers.RandomZoom(height_factor=(-0.18, 0.08), fill_mode="reflect"),
    layers.RandomBrightness(0.16, value_range=(0, 255)),
    layers.RandomContrast(0.18),
])

def build_dataset(df_sub, training=False):
    rutas = df_sub["ruta"].values
    labels_all = df_sub["label"].values
    if training:
        wild_mask = (df_sub["origen"] == "wild").values
        rutas_wild = rutas[wild_mask]
        labels_wild = labels_all[wild_mask]
        # Triplicar presencia de muestras wild para dominar invarianza de dominio
        rutas = np.concatenate([rutas, rutas_wild, rutas_wild])
        labels_all = np.concatenate([labels_all, labels_wild, labels_wild])

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
df_test_wild = df_test[df_test["origen"] == "wild"].reset_index(drop=True)
test_wild_ds = build_dataset(df_test_wild, training=False)

def confusion_aware_focal_loss(gamma=2.0, label_smoothing=0.03):
    def loss_fn(y_true, y_pred):
        y_true_smooth = y_true * (1.0 - label_smoothing) + (label_smoothing / float(NUM_CLASSES))
        y_pred_clip = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_smooth * tf.math.log(y_pred_clip)
        focal_weight = tf.pow(1.0 - y_pred_clip, gamma)
        return tf.reduce_sum(focal_weight * cross_entropy * cw_vector[None, :], axis=-1)
    return loss_fn

path_focal = os.path.join(MODELOS_DIR, "modelo_clasificador_19_clases_focal_loss.keras")
path_base = os.path.join(MODELOS_DIR, "modelo_clasificador_19_clases_efficientnetv2b0.keras")

print("🚀 Etapa 2: Descongelando backbone completo (excepto BatchNorm) con LR suave...")
model_focal = keras.models.load_model(path_focal, compile=False)
for layer in model_focal.layers:
    if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
        layer.trainable = True
        for sublayer in layer.layers:
            sublayer.trainable = not isinstance(sublayer, layers.BatchNormalization)
    elif not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

EPOCHS = 5
steps = len(train_ds)
lr_sched = keras.optimizers.schedules.CosineDecay(8e-5, EPOCHS * steps, alpha=0.05)
model_focal.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4),
    loss=confusion_aware_focal_loss(gamma=2.0, label_smoothing=0.03),
    metrics=["accuracy"]
)
model_focal.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
model_focal.save(path_focal)

print("🚀 Etapa 2 en model_base (3 épocas)...")
model_base = keras.models.load_model(path_base, compile=False)
for layer in model_base.layers:
    if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
        layer.trainable = True
        for sublayer in layer.layers:
            sublayer.trainable = not isinstance(sublayer, layers.BatchNormalization)
    elif not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

lr_sched_b = keras.optimizers.schedules.CosineDecay(7e-5, 3 * steps, alpha=0.05)
model_base.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=lr_sched_b, weight_decay=1e-4),
    loss=confusion_aware_focal_loss(gamma=1.5, label_smoothing=0.04),
    metrics=["accuracy"]
)
model_base.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)
model_base.save(path_base)

# Evaluación final
p_f = model_focal.predict(test_ds, verbose=0)
p_b = model_base.predict(test_ds, verbose=0)
p_ens = 0.80 * p_f + 0.20 * p_b
y_true_test = df_test["label"].values
y_pred_test = np.argmax(p_ens, axis=1)
print(f"\n🌟 Exactitud Test Global (19 clases): {accuracy_score(y_true_test, y_pred_test)*100:.2f}% | Macro F1: {f1_score(y_true_test, y_pred_test, average='macro'):.4f}")

p_fw = model_focal.predict(test_wild_ds, verbose=0)
p_bw = model_base.predict(test_wild_ds, verbose=0)
p_ens_w = 0.80 * p_fw + 0.20 * p_bw
y_true_w = df_test_wild["label"].values
y_pred_w = np.argmax(p_ens_w, axis=1)
print(f"🔥 Exactitud Test In-The-Wild (9 clases): {accuracy_score(y_true_w, y_pred_w)*100:.2f}%")
present = sorted(list(set(y_true_w)))
print(classification_report(y_true_w, y_pred_w, labels=present, target_names=[class_names[i] for i in present], digits=4))

# Exportación limpia a TFLite FP16 reconstruyendo en float32 puro
print("\n📱 Reconstruyendo grafo en float32 puro para exportación TFLite FP16...")
keras.mixed_precision.set_global_policy("float32")
inputs = keras.Input(shape=(180, 180, 3), dtype="float32", name="entrada_imagen_180x180")
base_f32 = keras.applications.EfficientNetV2B0(include_top=False, weights=None, input_shape=(180, 180, 3))
x = base_f32(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.BatchNormalization(name="bn_head")(x)
x = layers.Dropout(0.35, name="drop_1")(x)
x = layers.Dense(256, activation="swish", name="dense_256")(x)
x = layers.BatchNormalization(name="bn_dense")(x)
x = layers.Dropout(0.25, name="drop_2")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="clasificador_19_clases")(x)
export_model = keras.Model(inputs=inputs, outputs=outputs)

for dst_layer, src_layer in zip(export_model.layers, model_focal.layers):
    try:
        dst_layer.set_weights(src_layer.get_weights())
    except Exception:
        pass

converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
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
    print(f"✅ TFLite FP16 exportado con éxito a {tflite_path_main} y {tflite_flutter_path}")

