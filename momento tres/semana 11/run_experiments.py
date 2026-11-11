import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# 1. Configuración de Hardware y Memoria
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth habilitado.")
    except RuntimeError as e:
        print("Error en memory growth:", e)

mixed_precision.set_global_policy("mixed_float16")
print(f"Política de precisión: {mixed_precision.global_policy()}")

# 2. Parámetros del Experimento
DATASET_DIR = "dataset_completo"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE

# 3. Preparación de Datasets (70% Train, 15% Val, 15% Test)
print("\n--- Cargando Datasets ---")
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_test_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_raw.class_names
num_classes = len(class_names)
total_val_test = len(val_test_raw)
val_batches = total_val_test // 2

val_raw = val_test_raw.take(val_batches)
test_raw = val_test_raw.skip(val_batches)

print(f"Clases ({num_classes}): {class_names}")
print(f"Batches: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")

# Data Augmentation integrado
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Extracción de etiquetas verdaderas del conjunto de prueba independiente
print("\nPreparando etiquetas del test set independiente...")
y_test_batches = []
for images, labels in test_raw:
    y_test_batches.append(labels.numpy())
y_test = np.argmax(np.concatenate(y_test_batches, axis=0), axis=1)
print(f"Total muestras en prueba: {len(y_test)}")

# Optimización del pipeline tf.data
train_ds = train_raw.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_ds = val_raw.cache().prefetch(AUTOTUNE)
test_ds = test_raw.cache().prefetch(AUTOTUNE)

results = {}

# ==========================================
# EXPERIMENTO 1: RESNET50 (Pesada - Residual)
# ==========================================
print("\n" + "="*50)
print("INICIANDO EXPERIMENTO 1: ResNet50 (Arquitectura Pesada)")
print("="*50)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess

# Fase 1: Feature Extraction
inputs_resnet = layers.Input(shape=(180, 180, 3), name="input_resnet")
x = layers.Lambda(resnet_preprocess, name="resnet_preprocess")(inputs_resnet)
base_resnet = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
base_resnet.trainable = False

x = base_resnet.output
x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization(name="head_bn")(x)
x = layers.Dropout(0.3, name="head_dropout")(x)
x = layers.Dense(256, activation="relu", name="head_dense")(x)
x = layers.Dropout(0.2, name="head_dropout2")(x)
outputs_resnet = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

model_resnet = models.Model(inputs=inputs_resnet, outputs=outputs_resnet, name="ResNet50_Classifier")

model_resnet.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"ResNet50 - Parámetros totales: {model_resnet.count_params():,}")
print(f"ResNet50 - Parámetros entrenables (Fase 1): {sum(np.prod(p.shape) for p in model_resnet.trainable_weights):,}")

early_stop_phase1 = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

start_time = time.time()
print("\n--- ResNet50: Fase 1 (Extracción de Características, lr=1e-3) ---")
h1_resnet = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stop_phase1]
)
phase1_time_res = time.time() - start_time

# Fase 2: Fine-Tuning
print("\n--- ResNet50: Fase 2 (Fine-Tuning, lr=1e-5, BatchNormalization freeze) ---")
base_resnet.trainable = True
# Descongelar sólo las últimas capas (últimos 20 layers de ResNet50)
for layer in base_resnet.layers[:-20]:
    layer.trainable = False
# Regla de oro: congelar BatchNorm para no destruir estadísticas de ImageNet
for layer in base_resnet.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

print(f"ResNet50 - Parámetros entrenables (Fase 2): {sum(np.prod(p.shape) for p in model_resnet.trainable_weights):,}")

model_resnet.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb_phase2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

start_time_ft = time.time()
h2_resnet = model_resnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=cb_phase2
)
phase2_time_res = time.time() - start_time_ft
total_time_resnet = phase1_time_res + phase2_time_res

# Evaluación en Test Set Independiente
print("\nEvaluando ResNet50 en conjunto de prueba independiente...")
y_pred_probs_resnet = model_resnet.predict(test_ds)
y_pred_resnet = np.argmax(y_pred_probs_resnet, axis=1)

acc_resnet = float(accuracy_score(y_test, y_pred_resnet))
f1_macro_resnet = float(f1_score(y_test, y_pred_resnet, average="macro"))
cm_resnet = confusion_matrix(y_test, y_pred_resnet).tolist()

# Guardar modelo y medir tamaño
resnet_path = "momento tres/semana 11/modelo_resnet50_finetuned.keras"
model_resnet.save(resnet_path)
size_mb_resnet = round(os.path.getsize(resnet_path) / (1024 * 1024), 2)

# Benchmark de latencia de inferencia (100 imágenes)
warmup = model_resnet.predict(test_raw.take(1))
lat_start = time.time()
for _ in range(5):
    _ = model_resnet.predict(test_raw.take(2), verbose=0)
latency_ms_resnet = round(((time.time() - lat_start) / 10) * 1000 / 32, 2)

print(f"ResNet50 - Test Accuracy: {acc_resnet*100:.2f}% | Macro F1: {f1_macro_resnet*100:.2f}%")
print(f"Distancia a meta 94%: {max(0.0, 94.0 - acc_resnet*100):.2f}%")

# Guardar historial
results["ResNet50"] = {
    "total_params": int(model_resnet.count_params()),
    "trainable_params_phase1": int(sum(np.prod(p.shape) for p in h1_resnet.model.trainable_weights)),
    "size_mb": size_mb_resnet,
    "phase1_time_sec": round(phase1_time_res, 1),
    "phase2_time_sec": round(phase2_time_res, 1),
    "total_train_time_sec": round(total_time_resnet, 1),
    "latency_ms_per_image": latency_ms_resnet,
    "test_accuracy": acc_resnet,
    "test_f1_macro": f1_macro_resnet,
    "gap_to_94": round(max(0.0, 0.94 - acc_resnet), 4),
    "confusion_matrix": cm_resnet,
    "history_phase1": {k: [float(v) for v in vals] for k, vals in h1_resnet.history.items()},
    "history_phase2": {k: [float(v) for v in vals] for k, vals in h2_resnet.history.items()}
}

del model_resnet
tf.keras.backend.clear_session()

# ==========================================
# EXPERIMENTO 2: MOBILENETV2 (Liviana - DW Convs)
# ==========================================
print("\n" + "="*50)
print("INICIANDO EXPERIMENTO 2: MobileNetV2 (Arquitectura Liviana)")
print("="*50)

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

# Fase 1: Feature Extraction
inputs_mob = layers.Input(shape=(180, 180, 3), name="input_mobilenet")
x = layers.Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(inputs_mob)
base_mob = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
base_mob.trainable = False

x = base_mob.output
x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization(name="head_bn")(x)
x = layers.Dropout(0.3, name="head_dropout")(x)
x = layers.Dense(256, activation="relu", name="head_dense")(x)
x = layers.Dropout(0.2, name="head_dropout2")(x)
outputs_mob = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

model_mob = models.Model(inputs=inputs_mob, outputs=outputs_mob, name="MobileNetV2_Classifier")

model_mob.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"MobileNetV2 - Parámetros totales: {model_mob.count_params():,}")
print(f"MobileNetV2 - Parámetros entrenables (Fase 1): {sum(np.prod(p.shape) for p in model_mob.trainable_weights):,}")

early_stop_phase1 = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

start_time = time.time()
print("\n--- MobileNetV2: Fase 1 (Extracción de Características, lr=1e-3) ---")
h1_mob = model_mob.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stop_phase1]
)
phase1_time_mob = time.time() - start_time

# Fase 2: Fine-Tuning
print("\n--- MobileNetV2: Fase 2 (Fine-Tuning, lr=1e-5, BatchNormalization freeze) ---")
base_mob.trainable = True
# Descongelar últimas capas (últimos 35 layers de MobileNetV2)
for layer in base_mob.layers[:-35]:
    layer.trainable = False
# Congelar BatchNorm para inferencia
for layer in base_mob.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

print(f"MobileNetV2 - Parámetros entrenables (Fase 2): {sum(np.prod(p.shape) for p in model_mob.trainable_weights):,}")

model_mob.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cb_phase2 = [
    callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

start_time_ft = time.time()
h2_mob = model_mob.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=cb_phase2
)
phase2_time_mob = time.time() - start_time_ft
total_time_mob = phase1_time_mob + phase2_time_mob

# Evaluación en Test Set Independiente
print("\nEvaluando MobileNetV2 en conjunto de prueba independiente...")
y_pred_probs_mob = model_mob.predict(test_ds)
y_pred_mob = np.argmax(y_pred_probs_mob, axis=1)

acc_mob = float(accuracy_score(y_test, y_pred_mob))
f1_macro_mob = float(f1_score(y_test, y_pred_mob, average="macro"))
cm_mob = confusion_matrix(y_test, y_pred_mob).tolist()

# Guardar modelo y medir tamaño
mob_path = "momento tres/semana 11/modelo_mobilenetv2_finetuned.keras"
model_mob.save(mob_path)
size_mb_mob = round(os.path.getsize(mob_path) / (1024 * 1024), 2)

# Benchmark de latencia de inferencia
warmup = model_mob.predict(test_raw.take(1))
lat_start = time.time()
for _ in range(5):
    _ = model_mob.predict(test_raw.take(2), verbose=0)
latency_ms_mob = round(((time.time() - lat_start) / 10) * 1000 / 32, 2)

print(f"MobileNetV2 - Test Accuracy: {acc_mob*100:.2f}% | Macro F1: {f1_macro_mob*100:.2f}%")
print(f"Distancia a meta 94%: {max(0.0, 94.0 - acc_mob*100):.2f}%")

results["MobileNetV2"] = {
    "total_params": int(model_mob.count_params()),
    "trainable_params_phase1": int(sum(np.prod(p.shape) for p in h1_mob.model.trainable_weights)),
    "size_mb": size_mb_mob,
    "phase1_time_sec": round(phase1_time_mob, 1),
    "phase2_time_sec": round(phase2_time_mob, 1),
    "total_train_time_sec": round(total_time_mob, 1),
    "latency_ms_per_image": latency_ms_mob,
    "test_accuracy": acc_mob,
    "test_f1_macro": f1_macro_mob,
    "gap_to_94": round(max(0.0, 0.94 - acc_mob), 4),
    "confusion_matrix": cm_mob,
    "history_phase1": {k: [float(v) for v in vals] for k, vals in h1_mob.history.items()},
    "history_phase2": {k: [float(v) for v in vals] for k, vals in h2_mob.history.items()}
}

results["class_names"] = class_names

# Guardar resultados consolidados
with open("momento tres/semana 11/resultados_comparativa_arquitecturas.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "="*50)
print("¡TODOS LOS EXPERIMENTOS FINALIZADOS EXITOSAMENTE!")
print("Resultados exportados a: momento tres/semana 11/resultados_comparativa_arquitecturas.json")
print("="*50)
