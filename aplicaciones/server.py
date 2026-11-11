import os
import sys
import json
import time
import base64
import urllib.parse
import numpy as np

# Silenciar logs verbosos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

from http.server import HTTPServer, BaseHTTPRequestHandler

# 1. Definición exacta de las 19 clases y nombres legibles en español
class_names_19 = [
    'Aceites_Quimicos', 'Bombillas', 'Cardboard', 'Escombros_Ceramica',
    'Food_Organics', 'Glass', 'Madera_Procesada', 'Medicamentos',
    'Metal', 'Miscellaneous_Trash', 'Neumaticos_Gomas', 'Paper',
    'Pilas_Baterias', 'Plastic', 'RAEE_Electronicos', 'Residuos_Sanitarios',
    'Tetra_Pack', 'Textile_Trash', 'Vegetation'
]

NOMBRES_ESPANOL = {
    'Aceites_Quimicos': 'Aceites y Químicos',
    'Bombillas': 'Bombillas (Mercurio / LED)',
    'Cardboard': 'Cartón (Cardboard)',
    'Escombros_Ceramica': 'Escombros y Cerámica',
    'Food_Organics': 'Residuos Orgánicos / Alimentos',
    'Glass': 'Vidrio (Glass)',
    'Madera_Procesada': 'Madera Procesada',
    'Medicamentos': 'Medicamentos Vencidos',
    'Metal': 'Metal',
    'Miscellaneous_Trash': 'Residuos Varios (Trash)',
    'Neumaticos_Gomas': 'Neumáticos y Gomas',
    'Paper': 'Papel (Paper)',
    'Pilas_Baterias': 'Pilas y Baterías',
    'Plastic': 'Plástico (Plastic)',
    'RAEE_Electronicos': 'RAEE (Electrónicos)',
    'Residuos_Sanitarios': 'Residuos Sanitarios',
    'Tetra_Pack': 'Tetra Pak / Envases Asépticos',
    'Textile_Trash': 'Textiles (Ropa / Telas)',
    'Vegetation': 'Vegetación / Jardinería'
}

# 2. Asignación oficial según Normativa Colombia (Resolución 2184 de 2019 y RESPEL)
CANECAS_COLOMBIA = {
    # ⚪ CANECA BLANCA: Residuos Aprovechables (Limpio y Seco)
    'Plastic': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Asegúrate de que la botella o envase esté limpio, vacío y seco (sin residuos de líquidos ni comida). Aplástalo para reducir su volumen.'
    },
    'Cardboard': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Cajas y cartones limpios y secos. Desarmar y aplanar las cajas para optimizar espacio. Si está contaminado con grasa o comida, va a la caneca negra.'
    },
    'Glass': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Botellas y frascos de vidrio limpios y sin quebrar. Retira tapas metálicas o plásticas (deposítalas por separado también en la caneca blanca).'
    },
    'Paper': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Hojas de archivo, cuadernos, periódicos y revistas limpios y libres de humedad. No arrugar en exceso.'
    },
    'Metal': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Latas de bebidas, conservas, tapas metálicas y latón. Enjuagar y escurrir previamente para evitar olores y fauna nociva.'
    },
    'Tetra_Pack': {
        'caneca': 'Blanca',
        'color_nombre': 'Blanca',
        'color_hex': '#FFFFFF',
        'color_accent': '#38BDF8',
        'badge_bg': 'rgba(255, 255, 255, 0.12)',
        'tipo': 'Aprovechable (Reciclable)',
        'icono': '⚪',
        'titulo': 'Caneca Blanca · Residuos Aprovechables',
        'accion': 'Depositar en CANECA BLANCA',
        'instrucciones': 'Envases asépticos de bebidas. Enjuagar con un poco de agua, escurrir, desdoblar las solapas laterales y aplanar el envase.'
    },

    # 🟢 CANECA VERDE: Residuos Orgánicos Aprovechables
    'Food_Organics': {
        'caneca': 'Verde',
        'color_nombre': 'Verde',
        'color_hex': '#10B981',
        'color_accent': '#34D399',
        'badge_bg': 'rgba(16, 185, 129, 0.2)',
        'tipo': 'Orgánico Aprovechable (Compostable)',
        'icono': '🟢',
        'titulo': 'Caneca Verde · Orgánicos Aprovechables',
        'accion': 'Depositar en CANECA VERDE',
        'instrucciones': 'Cáscaras de frutas y verduras, sobras de comida, granos de café y cáscaras de huevo. Apto para compostaje y producción de biofertilizantes.'
    },
    'Vegetation': {
        'caneca': 'Verde',
        'color_nombre': 'Verde',
        'color_hex': '#10B981',
        'color_accent': '#34D399',
        'badge_bg': 'rgba(16, 185, 129, 0.2)',
        'tipo': 'Orgánico Aprovechable (Restos Vegetales)',
        'icono': '🟢',
        'titulo': 'Caneca Verde · Orgánicos Aprovechables',
        'accion': 'Depositar en CANECA VERDE',
        'instrucciones': 'Pasto cortado, hojas secas, flores marchitas y ramas pequeñas de poda. No mezclar con bolsas plásticas, alambres ni piedras.'
    },

    # ⚫ CANECA NEGRA: Residuos No Aprovechables / Ordinarios
    'Miscellaneous_Trash': {
        'caneca': 'Negra',
        'color_nombre': 'Negra',
        'color_hex': '#64748B',
        'color_accent': '#94A3B8',
        'badge_bg': 'rgba(100, 116, 139, 0.25)',
        'tipo': 'No Aprovechable (Ordinario)',
        'icono': '⚫',
        'titulo': 'Caneca Negra · Residuos No Aprovechables',
        'accion': 'Depositar en CANECA NEGRA',
        'instrucciones': 'Papel higiénico, servilletas usadas, cartones impregnados de grasa, papeles encerados o plastificados y empaques sucios. Destino: Relleno sanitario.'
    },
    'Textile_Trash': {
        'caneca': 'Negra',
        'color_nombre': 'Negra',
        'color_hex': '#64748B',
        'color_accent': '#94A3B8',
        'badge_bg': 'rgba(100, 116, 139, 0.25)',
        'tipo': 'No Aprovechable / Textil Degradado',
        'icono': '⚫',
        'titulo': 'Caneca Negra · Residuos No Aprovechables',
        'accion': 'Depositar en CANECA NEGRA (o Donación si está en buen estado)',
        'instrucciones': 'Prendas y retazos rotos o muy manchados que no admiten reciclaje textil. Si la prenda está limpia y usable, priorizar siempre la donación.'
    },
    'Madera_Procesada': {
        'caneca': 'Negra',
        'color_nombre': 'Negra',
        'color_hex': '#64748B',
        'color_accent': '#94A3B8',
        'badge_bg': 'rgba(100, 116, 139, 0.25)',
        'tipo': 'No Aprovechable / Manejo Especial',
        'icono': '⚫',
        'titulo': 'Caneca Negra · Manejo Especial',
        'accion': 'Depositar en CANECA NEGRA (o Servicio de Voluminosos)',
        'instrucciones': 'Madera procesada tratada con barnices, pinturas, aglomerados o clavos. Para maderas grandes o muebles, coordinar servicio municipal de voluminosos.'
    },

    # 🔴 CANECA ROJA: Residuos Peligrosos, Posconsumo y Manejo Especial (RESPEL)
    'Pilas_Baterias': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Residuo Peligroso / Posconsumo',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Peligroso / Posconsumo',
        'accion': 'Depositar en CANECA ROJA o Punto "Pilas con el Ambiente"',
        'instrucciones': 'Contiene metales pesados altamente tóxicos (litio, mercurio, cadmio). ¡Nunca arrojar a la basura común! Llevar a buzones de recolección en centros comerciales.'
    },
    'Medicamentos': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Residuo Peligroso / Farmacéutico',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Farmacéutico Vencido',
        'accion': 'Depositar en CANECA ROJA o Contenedor "Punto Azul"',
        'instrucciones': 'Medicamentos vencidos, tabletas a medio consumir y sus empaques. Depositar en contenedores Punto Azul ubicados en droguerías y EPS. Nunca al inodoro.'
    },
    'Aceites_Quimicos': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Residuo Peligroso / Químico (RESPEL)',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Químicos y Aceites (RESPEL)',
        'accion': 'Depositar en CANECA ROJA / Punto Limpio de Aceites',
        'instrucciones': 'Aceites de cocina usados, lubricantes, solventes o químicos. 1 litro de aceite contamina 1.000 litros de agua. Envasar en botella plástica y entregar en punto limpio.'
    },
    'Bombillas': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Residuo Peligroso / Mercurio (Posconsumo)',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Posconsumo Iluminación',
        'accion': 'Depositar en CANECA ROJA o Contenedor "Lúmina"',
        'instrucciones': 'Tubos fluorescentes, bombillos ahorradores y LEDs contienen vapores de mercurio tóxico. Entregarlos intactos (sin quebrar) en buzones de recolección Lúmina.'
    },
    'RAEE_Electronicos': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Manejo Especial / Electrónico (RAEE)',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Residuos Electrónicos (RAEE)',
        'accion': 'Depositar en CANECA ROJA o Centro de Acopio RAEE',
        'instrucciones': 'Cables, cargadores, circuitos, celulares y pequeños electrodomésticos. Entregar en campañas de reciclaje electrónico autorizadas (EcoCómputo / Red Verde).'
    },
    'Residuos_Sanitarios': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Residuo Peligroso / Riesgo Biológico',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Riesgo Biológico / Sanitario',
        'accion': 'Depositar en CANECA ROJA / Bolsa Roja Sellada',
        'instrucciones': 'Gasas, apósitos, guantes con secreciones, jeringas u objetos con fluidos corporales. Exigen disposición en bolsa roja y tratamiento térmico especializado.'
    },
    'Escombros_Ceramica': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Manejo Especial (Residuos RCD)',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Manejo Especial (Escombros RCD)',
        'accion': 'Manejo Diferenciado (Ruta Municipal de Escombros)',
        'instrucciones': 'Cerámicas, ladrillos, baldosas o cemento. Está prohibido dejarlos en la calle o con basura ordinaria. Solicitar el servicio especial municipal de escombros.'
    },
    'Neumaticos_Gomas': {
        'caneca': 'Roja',
        'color_nombre': 'Roja',
        'color_hex': '#EF4444',
        'color_accent': '#F87171',
        'badge_bg': 'rgba(239, 68, 68, 0.2)',
        'tipo': 'Manejo Especial Posconsumo (NFU)',
        'icono': '🔴',
        'titulo': 'Caneca Roja · Posconsumo Llantas (NFU)',
        'accion': 'Punto Posconsumo Especial ("Rueda Verde" / Servitecas)',
        'instrucciones': 'Neumáticos fuera de uso de bicicletas, motos o autos. Entregar en servitecas o puntos del programa posconsumo Rueda Verde para su aprovechamiento asfáltico.'
    }
}

# 3. Carga en Memoria de los Modelos del Ensamble Calibrado (>92% Exactitud)
path_focal = os.path.join(BASE_DIR, 'modelos', 'modelo_clasificador_19_clases_focal_loss.keras')
path_base = os.path.join(BASE_DIR, 'modelos', 'modelo_clasificador_19_clases_efficientnetv2b0.keras')

print("⏳ Cargando modelos optimizados del Cuaderno 4 en memoria...")
t0 = time.time()
model_focal = keras.models.load_model(path_focal, compile=False)
model_base = keras.models.load_model(path_base, compile=False)
print(f"✅ Ensamble Calibrado cargado en {time.time() - t0:.2f}s")

# Warmup inicial de GPU/CPU
dummy = np.random.uniform(0, 255, (1, 180, 180, 3)).astype(np.float32)
_ = model_focal(dummy, training=False)
_ = model_base(dummy, training=False)
print("⚡ Warmup de inferencia completado")

# 4. Función de inferencia de alta precisión
def clasificar_imagen_bytes(img_bytes, usar_tta=True, temp=0.75):
    inicio = time.time()
    
    # Decodificar imagen
    img_raw = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    # Redimensionar preservando aspecto con padding si es rectangular para no distorsionar geometrías
    shape = tf.shape(img_raw)
    h, w = shape[0], shape[1]
    
    # Si la imagen tiene una relación de aspecto muy dispar (>1.2), redimensionar con pad para no aplastar el objeto
    ratio = tf.maximum(h, w) / tf.minimum(h, w)
    img_resized = tf.cond(
        ratio > 1.25,
        lambda: tf.image.resize_with_pad(img_raw, 180, 180),
        lambda: tf.image.resize(img_raw, (180, 180))
    )
    
    x_orig = tf.expand_dims(tf.cast(img_resized, tf.float32), 0)

    if usar_tta:
        # Multi-Angle TTA (original 50%, horizontal flip 30%, vertical flip 20%)
        x_fliph = tf.image.flip_left_right(x_orig)
        x_flipv = tf.image.flip_up_down(x_orig)
        
        p_f_orig = model_focal(x_orig, training=False).numpy()
        p_f_fliph = model_focal(x_fliph, training=False).numpy()
        p_f_flipv = model_focal(x_flipv, training=False).numpy()
        p_focal = (0.50 * p_f_orig) + (0.30 * p_f_fliph) + (0.20 * p_f_flipv)
        
        p_b_orig = model_base(x_orig, training=False).numpy()
        p_b_fliph = model_base(x_fliph, training=False).numpy()
        p_b_flipv = model_base(x_flipv, training=False).numpy()
        p_base = (0.50 * p_b_orig) + (0.30 * p_b_fliph) + (0.20 * p_b_flipv)
    else:
        p_focal = model_focal(x_orig, training=False).numpy()
        p_base = model_base(x_orig, training=False).numpy()

    # Fusión ponderada del ensamble: 80% Focal + 20% Base
    p_ensamble = (0.80 * p_focal) + (0.20 * p_base)

    # Calibración de temperatura (Temperature Scaling, Guo et al. 2017)
    eps = 1e-7
    t_val = max(0.2, float(temp))
    log_p = np.log(np.clip(p_ensamble, eps, 1.0 - eps)) / t_val
    exp_p = np.exp(log_p - np.max(log_p))
    probs = (exp_p / np.sum(exp_p))[0]

    latencia_ms = round((time.time() - inicio) * 1000, 1)

    top_indices = probs.argsort()[-5:][::-1]
    top_predictions = []
    for idx in top_indices:
        c_name = class_names_19[idx]
        caneca_info = CANECAS_COLOMBIA.get(c_name, {
            'caneca': 'Negra',
            'color_nombre': 'Negra',
            'color_hex': '#64748B',
            'color_accent': '#94A3B8',
            'badge_bg': 'rgba(100, 116, 139, 0.25)',
            'tipo': 'No Aprovechable',
            'icono': '⚫',
            'titulo': 'Caneca Negra · Residuos No Aprovechables',
            'accion': 'Depositar en CANECA NEGRA',
            'instrucciones': 'Disponer con la recolección ordinaria de residuos no aprovechables.'
        })
        top_predictions.append({
            "label": c_name,
            "label_es": NOMBRES_ESPANOL.get(c_name, c_name),
            "conf": float(probs[idx]),
            "caneca": caneca_info
        })

    return {
        "success": True,
        "latency_ms": latencia_ms,
        "predictions": top_predictions,
        "caneca_sugerida": top_predictions[0]["caneca"] if top_predictions else None
    }

# 5. Servidor HTTP
class ServidorEcoClasificador(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        url_path = urllib.parse.urlparse(self.path).path
        if url_path == '/' or url_path == '/index.html':
            if os.path.exists(HTML_FILE):
                with open(HTML_FILE, 'rb') as fh:
                    content = fh.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html no encontrado")
        elif url_path == '/descargar-apk' or url_path == '/EcoClasificador_Colombia_Res2184.apk':
            apk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EcoClasificador_Colombia_Res2184.apk')
            if os.path.exists(apk_path):
                file_size = os.path.getsize(apk_path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.android.package-archive')
                self.send_header('Content-Disposition', 'attachment; filename="EcoClasificador_Colombia_Res2184.apk"')
                self.send_header('Content-Length', str(file_size))
                self.end_headers()
                with open(apk_path, 'rb') as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            else:
                self.send_error(404, "APK no encontrado")
        elif url_path.startswith('/dataset_completo/'):
            rel_path = urllib.parse.unquote(url_path[1:])
            abs_path = os.path.join(BASE_DIR, rel_path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                with open(abs_path, 'rb') as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, "Archivo no encontrado")
        else:
            self.send_error(404, "Ruta no encontrada")

    def do_POST(self):
        url_path = urllib.parse.urlparse(self.path).path
        if url_path == '/api/clasificar':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode('utf-8'))

                img_b64 = payload.get('image', '')
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                
                img_bytes = base64.b64decode(img_b64)
                usar_tta = payload.get('usar_tta', True)
                temp = payload.get('temperatura', 0.75)

                resultado = clasificar_imagen_bytes(img_bytes, usar_tta=usar_tta, temp=temp)

                res_bytes = json.dumps(resultado, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)

            except Exception as e:
                err_bytes = json.dumps({"success": False, "error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(404, "Endpoint no encontrado")

def run(puerto=7865):
    while puerto < 7885:
        try:
            httpd = HTTPServer(('127.0.0.1', puerto), ServidorEcoClasificador)
            print(f"\n=======================================================")
            print(f"🚀 EcoClasificador Web Oficial activo")
            print(f"🌐 URL Local: http://127.0.0.1:{puerto}")
            print(f"🎯 Motor: Ensamble Calibrado (Focal 80% + Base 20%)")
            print(f"🇨🇴 Normativa Colombia: Canecas Blanca, Negra, Verde y Roja")
            print(f"✨ Exactitud: 92.49% | Top-2: 98.24% | TTA Activado")
            print(f"=======================================================\n")
            httpd.serve_forever()
            break
        except OSError:
            puerto += 1

if __name__ == '__main__':
    run(7865)
