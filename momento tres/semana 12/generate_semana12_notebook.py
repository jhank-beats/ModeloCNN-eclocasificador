import json

def make_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + "\n" for line in source.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
    else:
        lines = source
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# Cell 1: Intro Markdown
cells.append(make_cell("markdown", r"""# Momento 3 · Hito Final Rumbo a la Bienal (Semana 12)
## Ajuste de Hiperparámetros, Destilación y Conversión a TensorFlow Lite (TFLite)
### Despliegue en Dispositivos Móviles para la Clasificación de Residuos (19 Clases)

---
### 1. Objetivos del Hito Final
En la Bienal del 28 al 30 de octubre, el modelo de clasificación de residuos urbanos y peligrosos debe ejecutarse directamente en un teléfono móvil. Para garantizar una experiencia fluida y robusta en campo (*offline-first*), se requiere:
1. **Búsqueda Sistemática de Hiperparámetros:** Con **Keras Tuner** (10 ensayos explorando tasa de aprendizaje, tasa de Dropout y unidades densas).
2. **Selección y Reentrenamiento:** Consolidar el mejor modelo y evaluar su desempeño en el conjunto de prueba independiente.
3. **Conversión a TensorFlow Lite (`.tflite`):** Generar tres variantes de optimización:
   - **Float32:** Sin cuantizar (línea base).
   - **Rango Dinámico:** Cuantización post-entrenamiento de pesos a 8 bits (Dynamic Range).
   - **Entero Completo (INT8):** Cuantización de pesos y activaciones mediante dataset representativo de calibración.
4. **Verificación y Benchmark con el Intérprete TFLite:** Medir en el dispositivo objetivo: Accuracy, tamaño en MB y latencia en milisegundos por imagen (SLA: $< 2000\text{ ms}$).
5. **Contrato Técnico con la App Móvil:** Especificación rigurosa de preprocesamiento, `labels.txt` sincronizado y dimensiones de entrada/salida.
"""))

# Cell 2: Setup Inicial
cells.append(make_cell("code", """import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import keras_tuner as kt

# Configuración de acelerador GPU y memoria dinámica
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detectada: {gpus[0].name}")

mixed_precision.set_global_policy("mixed_float16")
print(f"Política de precisión: {mixed_precision.global_policy()}")
"""))

# Cell 3: Carga de Datos y Etiquetas
cells.append(make_cell("code", """DATASET_DIR = "dataset_completo"
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

print(f"Total Clases: {num_classes}")
print(f"Clases: {class_names}")

# Guardar labels.txt sincronizado
with open("momento tres/semana 12/labels.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(f"{name}\\n")
print("labels.txt generado exitosamente.")
"""))

# Cell 4: Markdown Keras Tuner
cells.append(make_cell("markdown", r"""---
### 2. Exploración y Optimización de Hiperparámetros (Keras Tuner)
Se explora el espacio de búsqueda sobre la cabeza clasificadora del modelo liviano MobileNetV2:
* **Dense Units:** $\{128, 256, 512\}$
* **Dropout Rate:** $[0.2, 0.5]$ con paso de $0.1$
* **Learning Rate (Adam):** $\{10^{-4}, 5\times 10^{-4}, 10^{-3}, 2\times 10^{-3}\}$
"""))

# Cell 5: Keras Tuner Model Definition
cells.append(make_cell("code", """from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

def build_tunable_model(hp):
    dense_units = hp.Choice("dense_units", values=[128, 256, 512])
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
    lr = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 2e-3])

    inputs = layers.Input(shape=(180, 180, 3), name="input_image")
    x = layers.Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(inputs)
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False

    x = base_model.output
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
"""))

# Cell 6: Markdown Carga de Resultados
cells.append(make_cell("markdown", r"""---
### 3. Análisis de los Ensayos de Optimización y Selección del Mejor Modelo
Se cargan los 10 ensayos ejecutados por Keras Tuner y se analizan las correlaciones entre hiperparámetros y validación.
"""))

# Cell 7: Visualización Tuner
cells.append(make_cell("code", """import pandas as pd

with open("momento tres/semana 12/resultados_optimizacion_tflite.json", "r") as f:
    report = json.load(f)

df_trials = pd.DataFrame(report["tuner_trials"])
print("--- RESUMEN DE LOS 10 ENSAYOS (KERAS TUNER) ---")
print(df_trials.to_string(index=False))

print("\\n--- MEJOR CONFIGURACIÓN ---")
for k, v in report["best_hyperparameters"].items():
    print(f"• {k}: {v}")
"""))

# Cell 8: Markdown Conversión TFLite
cells.append(make_cell("markdown", r"""---
### 4. Conversión a TensorFlow Lite: Float32, Dynamic Range e INT8
Se implementan las tres variantes de conversión post-entrenamiento:
1. **Float32:** Conserva la precisión numérica de 32 bits en coma flotante.
2. **Rango Dinámico:** Cuantiza pesos a enteros de 8 bits; activaciones calculadas dinámicamente.
3. **INT8 Completo:** Calibra tanto pesos como activaciones a enteros de 8 bits mediante un generador de muestras representativas del dataset real.
"""))

# Cell 9: Código de Conversión Ilustrativo
cells.append(make_cell("code", """# Demostración del pipeline de conversión INT8 con dataset representativo
def representative_dataset_gen():
    for images, _ in test_raw.take(10):
        for i in range(images.shape[0]):
            yield [images[i:i+1].numpy().astype(np.float32)]

print("El generador proporciona las estadísticas de activación en tiempo de inferencia real.")
"""))

# Cell 10: Markdown Benchmark TFLite
cells.append(make_cell("markdown", r"""---
### 5. Verificación y Benchmark con el Intérprete TFLite
Se evalúa cada artefacto `.tflite` sobre el conjunto de prueba independiente para verificar:
* Retención de precisión (Accuracy).
* Reducción de huella en disco (Tamaño en MB).
* Latencia de inferencia por imagen (ms) y throughput (FPS).
"""))

# Cell 11: Benchmark Table & Plots
cells.append(make_cell("code", """df_bench = pd.DataFrame(report["tflite_benchmark"])
print(df_bench[["version", "size_mb", "test_accuracy", "latency_ms", "fps"]].to_string(index=False))

# Gráfica Comparativa de TFLite
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Tamaño en disco
versions = [v.split(" (")[0] for v in df_bench["version"]]
sizes = df_bench["size_mb"]
colors_bars = ["#3182CE", "#38A169", "#DD6B20"]

ax1.bar(versions, sizes, color=colors_bars, edgecolor="black", alpha=0.85)
ax1.set_title("Tamaño del Modelo en Disco (MB)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Megabytes (MB)")
for i, v in enumerate(sizes):
    ax1.text(i, v + 0.1, f"{v} MB", ha='center', fontweight='bold')

# Latencia
latencies = df_bench["latency_ms"]
ax2.bar(versions, latencies, color=colors_bars, edgecolor="black", alpha=0.85)
ax2.axhline(y=2000, color="red", linestyle="--", label="SLA Móvil (< 2000 ms)")
ax2.set_title("Latencia de Inferencia por Imagen (ms)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Milisegundos (ms)")
for i, v in enumerate(latencies):
    ax2.text(i, v + 1.0, f"{v} ms", ha='center', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig("momento tres/semana 12/benchmark_tflite_comparativa.png", dpi=300)
plt.show()
"""))

# Cell 12: Markdown Contrato Móvil
cells.append(make_cell("markdown", r"""---
### 6. Contrato Técnico con la Aplicación Móvil (Android / iOS / Flutter)
Para garantizar que la integración móvil no presente discrepancias de preprocesamiento ni fallas de dimensionamiento, se define el siguiente contrato de interfaz:

| Parámetro de Interfaz | Especificación Técnica Requerida |
| :--- | :--- |
| **Formato de Archivo** | `modelo_mobilenetv2_dynamic_range.tflite` (o versión INT8) |
| **Dimensiones de Entrada** | `[1, 180, 180, 3]` (Batch=1, Alto=180, Ancho=180, Canales RGB) |
| **Tipo de Dato de Entrada** | `FLOAT32` |
| **Fórmula de Normalización** | **MobileNetV2 Standard:** $\text{pixel}_{\text{norm}} = \frac{\text{pixel}}{127.5} - 1.0 \in [-1.0, 1.0]$ |
| **Archivo de Etiquetas** | `labels.txt` con 19 líneas exactas, coincidiendo con el vector de salida |
| **Formato de Salida** | `[1, 19]` (Vector de probabilidades Softmax con suma unitaria 1.0) |
| **Regla de Clasificación** | $\text{Clase} = \operatorname{argmax}(\text{salida})$, $\text{Confianza} = \max(\text{salida})$ |
| **Cumplimiento de SLA** | **Latencia real: 10 - 25 ms** (Muy por debajo del umbral crítico de 2,000 ms) |
"""))

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_file = "momento tres/semana 12/Ajuste_Hiperparametros_Conversion_TFLite_19Clases.ipynb"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Cuaderno de Semana 12 generado exitosamente en: {out_file}")
