import os
import time
import numpy as np
import tensorflow as tf
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_clasificador_19_clases.keras')

class_names_19 = [
    'Aceites_Quimicos', 'Bombillas', 'Carton', 'Escombros_Ceramica',
    'Madera_Procesada', 'Medicamentos', 'Metal', 'Neumaticos_Gomas',
    'Papel', 'Pilas_Baterias', 'Plastico', 'RAEE_Electronicos',
    'Residuos_Organicos', 'Residuos_Sanitarios', 'Residuos_Textiles',
    'Residuos_Varios', 'Tetra_Pack', 'Vegetacion', 'Vidrio'
]

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo de 19 clases cargado exitosamente.")
else:
    print("⚠️ Modelo no encontrado en ruta:", MODEL_PATH)

def clasificar_residuo(imagen):
    if imagen is None or model is None:
        return {"Modelo no disponible": 1.0}, "0.00 ms"
    img_resized = tf.image.resize(imagen, (224, 224))
    img_array = tf.expand_dims(img_resized, 0)
    inicio = time.time()
    preds = model.predict(img_array, verbose=0)[0]
    tiempo = (time.time() - inicio) * 1000
    resultados = {class_names_19[i]: float(preds[i]) for i in range(len(class_names_19))}
    return resultados, f"{tiempo:.2f} ms"

with gr.Blocks(title="♻️ Clasificador Inteligente de Residuos (19 Clases)") as demo:
    gr.Markdown("# ♻️ Clasificador Industrial de Residuos (19 Categorías)
Sube una imagen para clasificar el residuo con Inteligencia Artificial.")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Imagen del Residuo")
            btn = gr.Button("🔍 Clasificar Residuo", variant="primary")
        with gr.Column():
            output_labels = gr.Label(num_top_classes=5, label="Predicción y Probabilidades")
            output_time = gr.Textbox(label="Tiempo de Inferencia")
    btn.click(clasificar_residuo, inputs=[input_img], outputs=[output_labels, output_time])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860)
