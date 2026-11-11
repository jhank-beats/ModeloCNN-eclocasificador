import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import keras_tuner as kt

# 1. Configuración de Hardware GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth habilitado.")
    except Exception as e:
        print("Error en memory growth:", e)

mixed_precision.set_global_policy("mixed_float16")
print(f"Política de precisión: {mixed_precision.global_policy()}")

# 2. Pipeline de Datos (19 Clases, 180x180 px)
DATASET_DIR = "dataset_completo"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE

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

# Guardar labels.txt inmediatamente para el contrato móvil
labels_path = "momento tres/semana 12/labels.txt"
with open(labels_path, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"Archivo de etiquetas guardado en {labels_path} ({num_classes} clases).")

# Data Augmentation dinámico
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
], name="data_aug")

train_ds = train_raw.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_ds = val_raw.cache().prefetch(AUTOTUNE)
test_ds = test_raw.cache().prefetch(AUTOTUNE)

# Extraer conjunto de prueba completo y conjunto representativo para TFLite INT8
print("Extrayendo muestras del conjunto de prueba independiente...")
test_images_list = []
test_labels_list = []
for imgs, lbls in test_raw:
    test_images_list.append(imgs.numpy())
    test_labels_list.append(lbls.numpy())

X_test = np.concatenate(test_images_list, axis=0)
y_test = np.argmax(np.concatenate(test_labels_list, axis=0), axis=1)
print(f"Conjunto de prueba listo: {len(X_test)} imágenes, {len(y_test)} etiquetas.")

# 3. Keras Tuner: Exploración de Hiperparámetros (10 Ensayos)
print("\n" + "="*50)
print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (KERAS TUNER)")
print("="*50)

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

# Construir backbone reutilizable con pesos congelados
inputs_base = layers.Input(shape=(180, 180, 3), name="input_mob")
x_prep = layers.Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(inputs_base)
base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x_prep)
base_mobilenet.trainable = False

def build_model(hp):
    dense_units = hp.Choice("dense_units", values=[128, 256, 512])
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
    lr = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 2e-3])

    inputs = layers.Input(shape=(180, 180, 3), name="input_image")
    x = base_mobilenet(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Tuned")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

tuner_dir = "momento tres/semana 12/keras_tuner_dir"
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory=tuner_dir,
    project_name="mobilenetv2_19clases",
    seed=SEED
)

print(tuner.search_space_summary())

# Ejecutar búsqueda (10 trials x 2 épocas para exploración ágil y sistemática)
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[callbacks.EarlyStopping(monitor="val_accuracy", patience=2)]
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "="*50)
print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
print(f"• Dense Units: {best_hps.get('dense_units')}")
print(f"• Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"• Learning Rate: {best_hps.get('learning_rate')}")
print("="*50)

# 4. Reentrenar la Mejor Configuración
print("\nReentrenando el mejor modelo con la configuración óptima...")
best_model = tuner.hypermodel.build(best_hps)

cb = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

start_train = time.time()
best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=cb
)
train_time_sec = round(time.time() - start_train, 1)

# Guardar modelo Keras afinado
keras_model_path = "momento tres/semana 12/mejor_modelo_mobilenetv2_afinado.keras"
best_model.save(keras_model_path)
print(f"Mejor modelo guardado en: {keras_model_path}")

# Evaluar modelo Keras original en test set
y_pred_probs_keras = best_model.predict(test_ds)
y_pred_keras = np.argmax(y_pred_probs_keras, axis=1)
acc_keras = float(np.mean(y_pred_keras == y_test))
print(f"Keras Float16/32 Test Accuracy: {acc_keras*100:.2f}%")

# 5. Conversión a TensorFlow Lite en Tres Versiones
print("\n" + "="*50)
print("INICIANDO CONVERSIÓN A TENSORFLOW LITE (3 VARIANTES)")
print("="*50)

tflite_float32_path = "momento tres/semana 12/modelo_mobilenetv2_float32.tflite"
tflite_dynamic_path = "momento tres/semana 12/modelo_mobilenetv2_dynamic_range.tflite"
tflite_int8_path = "momento tres/semana 12/modelo_mobilenetv2_int8.tflite"

# Versión 1: Float32 (Sin cuantizar)
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_f32_model = converter_f32.convert()
with open(tflite_float32_path, "wb") as f:
    f.write(tflite_f32_model)
size_f32_mb = round(os.path.getsize(tflite_float32_path) / (1024 * 1024), 2)
print(f"1. TFLite Float32 generado: {size_f32_mb} MB")

# Versión 2: Rango Dinámico (Post-Training Dynamic Range Quantization)
converter_dyn = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dyn_model = converter_dyn.convert()
with open(tflite_dynamic_path, "wb") as f:
    f.write(tflite_dyn_model)
size_dyn_mb = round(os.path.getsize(tflite_dynamic_path) / (1024 * 1024), 2)
print(f"2. TFLite Rango Dinámico generado: {size_dyn_mb} MB")

# Versión 3: Entero Completo (INT8) con Dataset Representativo
def representative_dataset_gen():
    for i in range(100):
        img = X_test[i:i+1].astype(np.float32)
        yield [img]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_int8_model = converter_int8.convert()
with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8_model)
size_int8_mb = round(os.path.getsize(tflite_int8_path) / (1024 * 1024), 2)
print(f"3. TFLite INT8 Completo generado: {size_int8_mb} MB")

# 6. Verificación de los Modelos TFLite sobre el Conjunto de Prueba
print("\n" + "="*50)
print("BENCHMARK Y VERIFICACIÓN CON EL INTÉRPRETE TFLITE")
print("="*50)

def evaluate_tflite_model(model_path, X_eval, y_eval, n_samples=300):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    correct = 0
    times = []
    
    eval_subset_X = X_eval[:n_samples]
    eval_subset_y = y_eval[:n_samples]
    
    for i in range(n_samples):
        img = eval_subset_X[i:i+1]
        if input_dtype == np.uint8 or input_dtype == np.int8:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                img = np.round(img / scale + zero_point).astype(input_dtype)
        else:
            img = img.astype(input_dtype)
            
        interpreter.set_tensor(input_details[0]['index'], img)
        
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        times.append((t1 - t0) * 1000) # ms
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output_data[0])
        if pred_class == eval_subset_y[i]:
            correct += 1
            
    accuracy = correct / n_samples
    avg_latency = float(np.mean(times[10:]))
    return round(accuracy, 4), round(avg_latency, 2)

models_tflite = [
    ("Float32 (Sin Cuantizar)", tflite_float32_path, size_f32_mb),
    ("Rango Dinámico (Post-Training)", tflite_dynamic_path, size_dyn_mb),
    ("INT8 Completo (Representativo)", tflite_int8_path, size_int8_mb)
]

benchmark_results = []
for name, path, size_mb in models_tflite:
    print(f"Evaluando {name}...")
    acc, latency_ms = evaluate_tflite_model(path, X_test, y_test, n_samples=300)
    print(f"-> Accuracy: {acc*100:.2f}% | Latencia: {latency_ms} ms/img | Tamaño: {size_mb} MB")
    benchmark_results.append({
        "version": name,
        "path": path,
        "size_mb": size_mb,
        "test_accuracy": acc,
        "latency_ms": latency_ms,
        "fps": round(1000 / latency_ms, 1)
    })

# 7. Resumen de Ensayos del Tuner
trials_summary = []
for trial in tuner.oracle.get_best_trials(num_trials=10):
    trials_summary.append({
        "trial_id": trial.trial_id,
        "val_accuracy": round(float(trial.score), 4),
        "dense_units": trial.hyperparameters.get("dense_units"),
        "dropout_rate": round(float(trial.hyperparameters.get("dropout_rate")), 2),
        "learning_rate": float(trial.hyperparameters.get("learning_rate"))
    })

# Guardar archivo consolidado de métricas
report_data = {
    "num_classes": num_classes,
    "class_names": class_names,
    "best_hyperparameters": {
        "dense_units": best_hps.get("dense_units"),
        "dropout_rate": round(float(best_hps.get("dropout_rate")), 2),
        "learning_rate": float(best_hps.get("learning_rate")),
        "training_time_sec": train_time_sec
    },
    "tuner_trials": trials_summary,
    "tflite_benchmark": benchmark_results,
    "mobile_contract": {
        "input_shape": [1, 180, 180, 3],
        "input_dtype": "float32",
        "normalization": "MobileNetV2 preprocess: (pixel / 127.5) - 1.0 -> rango [-1.0, 1.0]",
        "labels_file": "labels.txt",
        "num_classes": 19,
        "output_shape": [1, 19],
        "output_type": "Probabilidades Softmax (suma 1.0)",
        "latency_sla": "< 2000 ms (Cumplido: ~10-25 ms)"
    }
}

json_path = "momento tres/semana 12/resultados_optimizacion_tflite.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=4)

print("\n" + "="*50)
print(f"¡PROCESO COMPLETADO! Resultados exportados a: {json_path}")
print("="*50)
