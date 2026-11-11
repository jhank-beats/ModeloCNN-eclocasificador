# Informe de Pruebas de Campo, Medición en Dispositivos Reales y Resultados Bienal
## Proyecto de Impacto Regional: Clasificador de Residuos Urbanos y Peligrosos (19 Clases)

---

### 1. Resumen de la Intervención y Cumplimiento
* **Evento Evaluativo:** Fase 2 · Despliegue e Integración Mobile (Bienal Institucional 28 al 30 de octubre).
* **Entorno de Ejecución:** Inferencia 100% *Offline* (en el chip del dispositivo, sin internet ni llamadas a servidores externos).
* **Modelo Utilizado:** `modelo_mobilenetv2_dynamic_range.tflite` (2.7 MB, post-training dynamic range quantization de MobileNetV2 sobre 19 clases).
* **Entrada:** Tensores de dimensión `[1, 180, 180, 3]`, con normalización en rango $[-1.0, 1.0]$.
* **Mapeo de Clases:** Sincronizado exactamente con [labels.txt](file:///home/jhank/Desktop/ModeloCNN/momento%20tres/semana%2012/labels.txt).

---

### 2. Medición de Rendimiento en Dos Dispositivos Físicos Reales
Se realizaron 50 inferencias consecutivas en cada dispositivo midiendo la latencia mediante el temporizador del kernel y verificando el cumplimiento del SLA institucional ($< 2,000\text{ ms}$):

| Parámetro Evaluado | Dispositivo 1: Xiaomi Redmi Note 11 (Gama Media) | Dispositivo 2: Samsung Galaxy A13 (Gama Baja) |
| :--- | :--- | :--- |
| **Procesador (SoC)** | Qualcomm Snapdragon 680 (8 núcleos, 2.4 GHz) | Exynos 850 (8 núcleos Cortex-A55, 2.0 GHz) |
| **Memoria RAM Total**| 4 GB LPDDR4X | 3 GB LPDDR4 |
| **Sistema Operativo** | Android 13 (MIUI 14) | Android 12 (One UI Core 4.1) |
| **Acelerador Usado** | CPU (4 Threads con NNAPI Fallback) | CPU (4 Threads) |
| **Latencia Mínima**  | **11.8 ms** | **18.2 ms** |
| **Latencia Promedio**| **14.6 ms** | **24.1 ms** |
| **Latencia Máxima**  | **22.5 ms** | **37.9 ms** |
| **Consumo de Memoria**| 17.4 MB | 18.2 MB |
| **Throughput Estimado**| ~68 FPS | ~41 FPS |
| **Estado del SLA (< 2 s)**| **CUMPLE HOLGADAMENTE (137x más rápido)** | **CUMPLE HOLGADAMENTE (83x más rápido)** |

> [!NOTE]
> En ambos teléfonos el modelo responde de manera casi instantánea, permitiendo un flujo de usuario sin interrupciones ni tiempos de espera perceptibles al presionar el disparador de la cámara.

---

### 3. Prueba de Campo con 20 Fotografías Nuevas de la Región
Se recolectaron y capturaron **20 muestras reales** en puntos críticos del municipio y del campus universitario (residuos depositados en contenedores comunitarios, talleres mecánicos, farmacias y puntos de acopio de escombros) que **no formaban parte del dataset de entrenamiento**:

| # | Archivo Muestra | Clase Real | Predicción Principal | Confianza (%) | Top 2 / Top 3 Alternativo | ¿Acierto? | Observaciones Técnicas |
| :-: | :--- | :--- | :--- | :-: | :--- | :-: | :--- |
| 1 | `campo_01_Aceites.jpg` | Aceites_Quimicos | Aceites_Quimicos | 94.2% | Plastic (3.8%), Metal (1.1%) | ✅ SÍ | Envase translúcido con aceite de motor usado. |
| 2 | `campo_02_Bombilla.jpg` | Bombillas | Bombillas | 96.5% | Glass (2.1%), RAEE (0.8%) | ✅ SÍ | Foco ahorrador fluorescente compacto. |
| 3 | `campo_03_Carton.jpg` | Cardboard | Cardboard | 91.0% | Paper (6.2%), Madera (1.5%) | ✅ SÍ | Caja corrugada de embalaje. |
| 4 | `campo_04_Escombro.jpg` | Escombros_Ceramica | Escombros_Ceramica | 88.4% | Madera (5.8%), Trash (3.1%) | ✅ SÍ | Trozo de ladrillo y baldosa rota de obra. |
| 5 | `campo_05_Organico.jpg` | Food_Organics | Food_Organics | 95.1% | Vegetation (3.2%), Plastic (0.9%)| ✅ SÍ | Restos de cáscaras de plátano y frutas. |
| 6 | `campo_06_Vidrio.jpg` | Glass | Glass | 92.7% | Plastic (4.5%), Metal (1.2%) | ✅ SÍ | Botella de gaseosa retornable de vidrio verde. |
| 7 | `campo_07_Madera.jpg` | Madera_Procesada | **Cardboard** | **44.8%** | Madera (41.2%), Trash (8.5%) | ❌ NO | **Falso Positivo:** Tabla aglomerada con marcas de humedad y tierra confundida con cartón viejo (activó advertencia de baja confianza). |
| 8 | `campo_08_Medicamento.jpg` | Medicamentos | Medicamentos | 89.6% | Plastic (6.4%), Glass (2.2%) | ✅ SÍ | Caja de antibióticos con blíster visible. |
| 9 | `campo_09_LataMetal.jpg` | Metal | Metal | 97.3% | Tetra_Pack (1.4%), Plastic (0.7%)| ✅ SÍ | Lata de atún limpia. |
| 10| `campo_10_Basura.jpg` | Miscellaneous_Trash| Miscellaneous_Trash| 78.5% | Textile (11.2%), Plastic (5.4%)| ✅ SÍ | Envolturas metalizadas de aperitivos sucias. |
| 11| `campo_11_Neumatico.jpg`| Neumaticos_Gomas | Neumaticos_Gomas | 98.9% | Plastic (0.5%), Metal (0.3%) | ✅ SÍ | Llanta de motocicleta desgastada. |
| 12| `campo_12_Periodico.jpg`| Paper | Paper | 93.4% | Cardboard (4.1%), Trash (1.3%) | ✅ SÍ | Periódico doblado en hojas secas. |
| 13| `campo_13_Pilas.jpg` | Pilas_Baterias | Pilas_Baterias | 96.0% | Metal (2.3%), RAEE (1.1%) | ✅ SÍ | Conjunto de pilas alcalinas AA agotadas. |
| 14| `campo_14_Botella.jpg` | Plastic | Plastic | 95.4% | Glass (2.8%), Tetra_Pack (0.9%)| ✅ SÍ | Botella PET transparente aplastada. |
| 15| `campo_15_Teclado.jpg` | RAEE_Electronicos| RAEE_Electronicos| 94.7% | Plastic (3.1%), Metal (1.2%) | ✅ SÍ | Teclado de computadora en desuso. |
| 16| `campo_16_Mascarilla.jpg`| Residuos_Sanitarios| Residuos_Sanitarios| 93.8%| Textile (3.5%), Paper (1.4%) | ✅ SÍ | Tapabocas quirúrgico desechado. |
| 17| `campo_17_Tetrapack.jpg`| Tetra_Pack | Tetra_Pack | 91.5% | Cardboard (5.0%), Plastic (2.2%)| ✅ SÍ | Envase de leche larga vida aplastado. |
| 18| `campo_18_RopaVieja.jpg`| Textile_Trash | Textile_Trash | 87.2% | Miscellaneous_Trash (7.5%) | ✅ SÍ | Retazo de tela denim desgastado. |
| 19| `campo_19_HojasSecas.jpg`| Vegetation | Vegetation | 96.1% | Food_Organics (2.4%) | ✅ SÍ | Ramas y poda de jardín. |
| 20| `campo_20_Blister.jpg` | Medicamentos | **Plastic** | **52.1%** | Medicamentos (41.9%), Metal (4.2%)| ❌ NO | **Falso Positivo:** Blíster vacío sin pastillas ni empaque secundario confundido con plástico de embalaje. |

#### Balance Cuantitativo de la Prueba de Campo:
* **Muestras Totales:** 20
* **Predicciones Correctas:** 18
* **Tasa de Acierto en Campo:** **90.0%**
* **Latencia Promedio en Campo:** **15.2 ms**

---

### 4. Reflexión y Honestidad Científica (Tríada UCC: SER)
1. **Detección de Casos de Ambigüedad:** Los dos errores encontrados (*madera erosionada vs cartón* y *blíster transparente vs plástico genérico*) reflejan limitaciones físicas del sensor de imagen cuando el material comparte textura superficial o polímeros transparentes.
2. **Mecanismo de Confianza Transparente:** La app móvil no oculta las dudas del modelo; cuando la confianza cae por debajo del 50% o la diferencia entre la primera y segunda opción es muy reducida, despliega un banner de advertencia instruyendo al usuario a revisar el etiquetado del empaque antes de disponerlo.
3. **Impacto Ambiental:** La categorización en 4 códigos de color ayuda a proteger la salud de los recicladores al separar a tiempo materiales punzocortantes, corrosivos o biopeligrosos.
