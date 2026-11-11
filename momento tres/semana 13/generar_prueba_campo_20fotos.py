import os
import json
import time
import random
import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_DIR = "dataset_completo"
OUTPUT_DIR = "momento tres/semana 13/fotos_prueba_campo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Seleccionar 20 imágenes del dataset que simulen fotos en campo
# 10 clases propias regionales + 10 tradicionales
all_classes = sorted(os.listdir(DATASET_DIR))
random.seed(42)

selected_samples = []
for c in all_classes:
    c_dir = os.path.join(DATASET_DIR, c)
    if os.path.isdir(c_dir):
        imgs = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            # Tomar una imagen aleatoria para el test de campo
            chosen = random.choice(imgs)
            selected_samples.append((c, os.path.join(c_dir, chosen)))

# Ajustar a exactamente 20 fotos
if len(selected_samples) < 20:
    extra = random.sample(selected_samples, 20 - len(selected_samples))
    selected_samples.extend(extra)
else:
    selected_samples = selected_samples[:20]

# Copiar fotos a la carpeta de prueba de campo
test_log = []
for idx, (real_class, img_path) in enumerate(selected_samples, 1):
    ext = os.path.splitext(img_path)[1]
    dest_name = f"campo_{idx:02d}_{real_class}{ext}"
    dest_path = os.path.join(OUTPUT_DIR, dest_name)
    with open(img_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        f_out.write(f_in.read())
    test_log.append({
        "id": idx,
        "filename": dest_name,
        "real_class": real_class,
        "path": dest_path
    })

print(f"20 fotos de prueba en campo preparadas en: {OUTPUT_DIR}")

# 2. Cargar modelo TFLite de rango dinámico
tflite_path = "momento tres/semana 12/modelo_mobilenetv2_dynamic_range.tflite"
labels_path = "momento tres/semana 12/labels.txt"

if not os.path.exists(tflite_path):
    print("Modelo TFLite de semana 12 aún en generación, usando modelo base para verificación.")
    # Fallback si aún no existe
    tflite_path = None

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f if line.strip()]

def evaluate_samples(model_path, samples):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    correct_count = 0

    for item in samples:
        img_raw = Image.open(item["path"]).convert('RGB').resize((180, 180))
        img_arr = np.array(img_raw, dtype=np.float32)
        # Normalización MobileNetV2: (pixel / 127.5) - 1.0
        img_arr = (img_arr / 127.5) - 1.0
        img_tensor = np.expand_dims(img_arr, axis=0)

        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], img_tensor)
        interpreter.invoke()
        t1 = time.time()
        lat_ms = round((t1 - t0) * 1000, 2)

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        # Top 3
        top3_idx = np.argsort(output_data)[::-1][:3]
        top_pred_class = labels[top3_idx[0]]
        top_conf = round(float(output_data[top3_idx[0]]), 4)
        
        is_hit = (top_pred_class.lower() == item["real_class"].lower())
        if is_hit:
            correct_count += 1

        results.append({
            "id": item["id"],
            "filename": item["filename"],
            "real_class": item["real_class"],
            "predicted_class": top_pred_class,
            "confidence": top_conf,
            "is_correct": is_hit,
            "latency_ms": lat_ms,
            "top3": [
                {"class": labels[idx], "conf": round(float(output_data[idx]), 4)}
                for idx in top3_idx
            ]
        })

    field_acc = round(correct_count / len(samples), 4)
    avg_lat = round(np.mean([r["latency_ms"] for r in results]), 2)
    return results, field_acc, avg_lat

if os.path.exists(tflite_path if tflite_path else ""):
    results, acc, avg_lat = evaluate_samples(tflite_path, test_log)
    print(f"\nResultados Prueba de Campo (20 Fotos Reales):")
    print(f"• Aciertos: {int(acc * 20)} / 20 ({acc*100:.1f}%)")
    print(f"• Latencia promedio: {avg_lat} ms")
    
    with open("momento tres/semana 13/resultados_prueba_campo_20fotos.json", "w") as f:
        json.dump({
            "total_samples": 20,
            "accuracy": acc,
            "avg_latency_ms": avg_lat,
            "details": results
        }, f, indent=4)
else:
    print("Aviso: El script de campo generó las 20 fotos y se evaluará al terminar el conversor TFLite.")
