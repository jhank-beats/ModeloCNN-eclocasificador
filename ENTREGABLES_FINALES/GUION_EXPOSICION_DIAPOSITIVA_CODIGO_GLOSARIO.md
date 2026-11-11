# 🎙️ GUION DE EXPOSICIÓN FINAL — DIAPOSITIVA → CÓDIGO → GLOSARIO
## Proyecto: EcoClasificador Móvil con Visión Artificial y Edge AI (19 Clases)
### De 70.88 % a 92.49 % · Estructura oficial de sustentación

---

## 🧭 CÓMO FUNCIONA ESTA ESTRUCTURA

```
BLOQUE I    → Diapositivas 1–3  (apertura normal)
BLOQUE II   → CUADERNO 1: diapositivas 4–7  →  código notebook 1  →  glosario C1
BLOQUE III  → CUADERNO 2: diapositivas 8–9  →  código notebook 2  →  glosario C2
BLOQUE IV   → CUADERNO 3: diapositivas 10–11 →  código notebook 3  →  glosario C3
BLOQUE V    → CUADERNO 4: diapositivas 12–15 →  código notebook 4  →  glosario C4
BLOQUE VI   → Diapositivas 16–18 (despliegue y cierre)
```

> **Regla de la nueva estructura:** por cada cuaderno, primero **pasas TODAS sus diapositivas** (para que el jurado entienda la teoría), y **después abres el notebook** y muestras solo las celdas clave (2–4), cerrando con el **glosario y los conceptos clave** de ese cuaderno. Así conectas “lo que se dijo” con “lo que se programó”.

**Dos modos de tiempo:**
- **Ruta rápida (15–18 min):** muestra **2 celdas** por cuaderno.
- **Ruta completa (25–30 min):** muestra **3–4 celdas** por cuaderno + glosario.

---

# ⏱️ CRONOGRAMA

| Bloque | Contenido | Diapositivas | Código | Tiempo rápido | Tiempo completo |
| :--- | :--- | :---: | :---: | :---: | :---: |
| I | Apertura, problema, solución | 1–3 | — | 3 min | 3 min |
| II | Cuaderno 1 (dataset) | 4–7 | nb-1 | 5 min | 8 min |
| III | Cuaderno 2 (baseline) | 8–9 | nb-2 | 4 min | 6 min |
| IV | Cuaderno 3 (regularización) | 10–11 | nb-3 | 3 min | 5 min |
| V | Cuaderno 4 (optimización) | 12–15 | nb-4 | 5 min | 8 min |
| VI | Despliegue y cierre | 16–18 | — | 2 min | 2 min |

---

# 🟢 BLOQUE I — APERTURA (Diapositivas 1–3)

## 🔷 1 — Portada (45 s)
**Qué decir:** "Muy buenos días, profesor Óscar Díaz Triana… Soy Juan Camilo Murillo Moncaleano, de Electiva III. Sustento el **EcoClasificador Móvil**: 19 clases de residuos, del **70.88 % al 92.49 %**, en 6.4 MB y sin internet."

**📖 Qué es cada cosa:** CNN (red neuronal para imágenes) · Edge AI (inferencia local sin nube) · 19 clases (categorías de residuos).

---

## 🔷 2 — La Crisis de los Residuos (1 min)
**Qué se muestra:** El problema (17 % reciclado, rellenos colapsando, separación en la fuente) + panel de 10 clases peligrosas.

**🎤 Qué decir:** "En Colombia y Latinoamérica **menos del 17 %** se recicla; más del 80 % va a rellenos a punto de colapsar. La falla está **en la fuente**: el ciudadano no sabe separar residuos complejos. Y lo grave —panel derecho— es que residuos peligrosos (bombillas con mercurio, aceites, pilas, biosanitarios) contaminan el material reciclable y ponen en riesgo a los recicladores."

**📖 Qué es cada cosa:**
- **Residuo sólido urbano (RSU):** basura de casas y comercios.
- **Relleno sanitario:** vertedero controlado (Doña Juana, El Carrasco).
- **Separación en la fuente:** clasificar la basura donde se genera.
- **Residuo peligroso:** el que daña la salud o el ambiente (mercurio, aceites, Cd/Pb, biosanitarios).

---

## 🔷 3 — La Solución: EcoClasificador UCC (1 min)

**Qué se muestra:** 3 tarjetas (Edge AI / 19 clases / tiempo real) + diagrama metodológico.

**🎤 Qué decir:** "Nuestra respuesta es un ecosistema **Edge AI** con 3 pilares: corre local (6.4 MB, sin internet), 19 clases reales, y responde en **< 25 ms** indicando el contenedor normativo."

### 📖 Qué es cada cosa (definiciones que pediste)

**Resolución 2184 de 2019**
> Expedida por el **Ministerio de Ambiente y Desarrollo Sostenible de Colombia**, **unificó el código de colores** para la separación de residuos sólidos en la fuente a nivel nacional y **modificó disposiciones sobre el uso racional de bolsas plásticas**.
> - **Blanco** → aprovechables (reciclables).
> - **Negro** → no aprovechables (ordinarios).
> - **Verde** → orgánicos aprovechables.
> - **Rojo** → peligrosos (posconsumo).

**TFLite (TensorFlow Lite, ahora oficialmente LiteRT)**
> Es un **conjunto de herramientas de Google** que permite **ejecutar modelos de IA y Machine Learning directamente en dispositivos móviles, de borde (Edge) e Internet de las Cosas (IoT)**. En el proyecto: convierte el modelo Keras de 113 MB en un `.tflite` de **6.4 MB** que corre offline en el celular en **15–25 ms**.

**Otros términos de la lámina:**
- **Edge AI:** inferencia en el dispositivo, sin nube.
- **< 25 ms:** tiempo por imagen.
- **CRISP-ML(Q):** metodología de 6 fases con aseguramiento de calidad.

---

# 🟢 BLOQUE II — CUADERNO 1 (Dataset)

## 1.1 — Diapositivas del Cuaderno 1 (4–7)

### 🔷 4 — Recolección Guiada (Scraping) — 1 min
**🎤 Qué decir:** "Partimos de RealWaste (9 clases) y agregamos **10 clases críticas**. Recolectamos con un crawler de Bing con **35+ consultas semánticas por clase** en español e inglés y términos negativos (`-vector`, `-diagram`). Meta: **1,000 fotos limpias por clase**."

**📖 Qué es cada cosa:** Web scraping · Crawler (Bing Image Crawler) · Consulta semántica · Término negativo · RealWaste.

---

### 🔷 5 — Limpieza Automática — 1 min

**🎤 Qué decir:** "Cada imagen pasa un filtro estricto: corruptas, borrosas, sin contraste, infografías, dibujos y **duplicados por hash perceptual**. Luego limpieza anti-catálogo y revisión humana."

> ⚠️ **Aquí está la frase clave que pediste:** "No usamos MD5 para esto, usamos **hash perceptual (pHash)**: detecta imágenes *visualmente* parecidas aunque tengan distinta compresión."

### 📖 Qué es cada cosa (definiciones que pediste)

**Hash perceptual (pHash)**
> Es una **huella digital única que se genera a partir del contenido visual (o auditivo) de un archivo**. A diferencia de un hash criptográfico normal (que cambia con cualquier byte), el perceptual cambia **muy poco** si la imagen se comprime, se reescala o se le cambia el formato: por eso sirve para detectar “la misma imagen” aunque no sea idéntica byte a byte.

**Distancia de Hamming**
> Es la **métrica que se usa para comparar dos hashes perceptuales** y saber qué tan parecidos son:
> - **¿Qué mide?** Cuenta el **número de bits diferentes** entre las dos secuencias binarias.
> - **¿Qué es “Hamming 2”?** Significa que al comparar el hash de una imagen A con el de una imagen B, **solo hay 2 bits de diferencia** → se consideran casi la misma imagen (nuestro umbral fue **≤ 2**).

**Otros términos de la lámina:**
- **Varianza Laplaciana:** mide nitidez (baja = borrosa).
- **Filtro de calidad:** descarta corruptas, oscuras, quemadas, sin contraste.
- **Limpieza anti-catálogo:** descarta fotos con fondo blanco de estudio.
- **Depurador visual (Gradio):** revisión humana de un clic.

---

### 🔷 6 — Normalización con YOLOv8 ⭐ — 1.5 min

**🎤 Qué decir:** "El resize directo **deforma** (squashing) y el letterbox **añade bordes falsos**. Con **YOLOv8n** detectamos el residuo, expandimos su caja a **1:1 con 8 % de margen** y reescalamos a **180×180 con Lanczos**. Si no hay objeto único, usamos recorte central cuadrado."

**📖 Qué es cada cosa:** YOLOv8n · Bounding box `[x1,y1,x2,y2]` · Margen 8 % · Squashing · Letterbox · Lanczos · Center-crop (fallback) · Contrato 180×180×3.

---

### 🔷 7 — Auditoría y Partición — 1 min

**🎤 Qué decir:** "Cerramos el Cuaderno 1 con dos garantías de integridad."

### 📖 Qué es cada cosa (definiciones que pediste)

**Auditoría criptográfica (MD5)**
> **MD5** genera una **huella digital de 128 bits a partir de los bytes exactos** de un archivo. Si dos imágenes son idénticas byte a byte (aunque tengan nombre distinto), su MD5 es **igual**.
> - **“Elimina copias idénticas descargadas de distintos servidores”:** si el mismo archivo se descargó dos veces, se detecta por su MD5 y se borra una copia.

**Contradicciones inter-clase**
> Es cuando **una misma imagen aparece en dos carpetas/categorías distintas** (p. ej. una lata etiquetada como *Metal* y también como *RAEE*). El sistema la detecta y **la elimina automáticamente** para no enseñarle a la red señales contradictorias.

**Partición estratificada y persistente 70 / 15 / 15**
- **70 % Train** (7,966) → entrenar.
- **15 % Val** (1,707) → validar y ajustar hiperparámetros.
- **15 % Test** (1,707) → prueba final estrictamente reservada.
- **Estratificada:** cada partición conserva la proporción de las 19 clases.
- **Persistente:** se guarda en un CSV → **reproducibilidad total**.

> 🔎 **Dato clave para la defensa:** el **dHash (perceptual)** se usa en la limpieza rápida; el **MD5 (criptográfico)** se usa en la auditoría de integridad y contradicciones. **Son capas complementarias, no lo mismo.**

---

## 1.2 — Código en vivo: Cuaderno 1

> **Abre:** `momento_1_y_2_base/1_Pipeline_Dataset_10k_Imagenes.ipynb` (19 celdas).
> **Objetivo:** mostrar las 2–3 celdas que conectan con las diapositivas 4–7.

| Celda | Qué se ve | Qué decir |
| :--- | :--- | :--- |
| **[MD 3 / CODE 4]** | Lista `CATEGORIAS_10_CLASES` + consultas semánticas | "Aquí están las **10 clases** y las **35+ consultas** por clase, con filtros tipo `-vector` para evitar infografías." |
| **[MD 7 / CODE 8]** | `calcular_dhash`, `distancia_hamming`, `evaluar_calidad_imagen` | "Esta es la limpieza: **dHash** + **Hamming ≤ 2** para duplicados, y el filtro de **varianza Laplaciana** para borrosas." |
| **[MD 13 / CODE 14]** ⭐ | `recortar_y_normalizar_yolo` | "Y aquí está el **YOLOv8**: detecta la caja, la expande a 1:1 con **8 %** y reescala a **180×180 con Lanczos**." |
| **[MD 17 / CODE 18]** | Verificación de dimensiones | "Confirmamos que el **100 %** de las imágenes quedó en **180×180×3**." |

**Monólogo de cierre del código (30 s):**
> "Como ven, el notebook ejecuta exactamente lo que dice la diapositiva: primero define las 10 clases, luego filtra con dHash y Hamming, y al final recorta con YOLO a 180×180. El resultado es un dataset **limpio, sin duplicados y sin contradicciones**."

---

## 1.3 — Glosario y conceptos clave del Cuaderno 1

| Concepto | Definición corta |
| :--- | :--- |
| **Dataset** | Conjunto de imágenes etiquetadas para entrenar. |
| **Web scraping** | Descarga automática de imágenes desde buscadores. |
| **Crawler** | Programa que recorre resultados y descarga archivos. |
| **Consulta semántica** | Búsqueda con palabras precisas (ES/EN). |
| **pHash / dHash** | Huella digital **visual** de una imagen. |
| **Distancia de Hamming** | Nº de bits distintos entre dos hashes (≤ 2 = duplicada). |
| **MD5** | Huella criptográfica de **bytes** (copia exacta). |
| **Varianza Laplaciana** | Medida de nitidez. |
| **Squashing / Letterbox** | Deformación / franjas por resize ingenuo. |
| **Lanczos** | Interpolación de alta calidad. |
| **YOLOv8n** | Detector nano en tiempo real. |
| **Partición 70/15/15** | Train / Val / Test. |
| **Estratificada** | Conserva la proporción de clases. |

---

# 🟢 BLOQUE III — CUADERNO 2 (Baseline)

## 2.1 — Diapositivas del Cuaderno 2 (8–9)

### 🔷 8 — Modelo Padre MobileNetV2 — 1 min

**🎤 Qué decir:** "El Cuaderno 2 entrenó el **modelo padre** con **Transfer Learning** sobre MobileNetV2 pre-entrenado en ImageNet: entrada 180×180, **backbone congelado**, GAP → Dense(256) → Dropout → Softmax de 19. Entrenamos en **dos fases**: warmup y fine-tuning."

### 📖 Qué es cada cosa (definiciones que pediste)

**¿Qué es el Transfer Learning y cómo se aplicó?**
> Es el **aprendizaje por transferencia**: en vez de entrenar la red desde cero (lo que requiere millones de imágenes), **reutilizamos el conocimiento visual que MobileNetV2 ya aprendió en ImageNet** (bordes, texturas, formas) y solo le enseñamos la “capa final” para nuestras 19 clases.
> **Cómo se aplicó:**
> 1. Se carga MobileNetV2 con `weights="imagenet"` y `include_top=False` (sin su cabeza original de 1,000 clases).
> 2. El backbone se **congela** (`trainable=False`).
> 3. Se añade una **cabeza nueva**: GAP → Dense(256, ReLU) → Dropout → Dense(19, Softmax).
> 4. Se entrena la cabeza (warmup) y luego se **descongelan** las últimas capas (fine-tuning).

**¿Por qué MobileNetV2 y no MobileNetV1?**
> MobileNetV2 **mejora** a V1 con **residuales invertidos** (inverted residuals) y **cuellos lineales** (linear bottlenecks), que **evitan la pérdida de información** que produce ReLU en espacios comprimidos. Resultado: **misma eficiencia** de V1 (depthwise separable) pero **mayor precisión** con pocos parámetros (~2.5 M). V1 no tiene residuales ni cuellos lineales.

**Tu notación explicada, elemento por elemento:**
- **“Backbone MobileNetV2 pre-entrenado en ImageNet-1k”** → el tronco extractor de características, ya entrenado con 1,000 clases de ImageNet.
- **“Cabeza: GAP → Dense(256, ReLU) → Dropout(0.3) → Dense(19, Softmax)”**:
  - **GAP** = Global Average Pooling: promedia cada mapa de características → vector.
  - **Dense(256, ReLU)**: capa totalmente conectada de 256 neuronas con activación ReLU.
  - **Dropout(0.3)**: apaga 30 % de neuronas para no memorizar.
  - **Dense(19, Softmax)**: 19 neuronas (una por clase) que se convierten en probabilidades que suman 1.
- **“Input(180,180,3) → MobileNetV2 (congelado) → GAP → Dense(256) → Dropout(0.3) → Dense(19, Softmax)”** → el flujo completo: imagen 180×180×3 → extractor congelado → reducción → clasificación.

**Warmup y Fine-tuning (definición):**
- **Warmup:** 1.ª fase. El **backbone está congelado**; solo se entrena la cabeza nueva con un learning rate normal (Adam **1e-3**). Se llama “calentamiento” porque la cabeza aprende “desde cero” sin dañar el backbone.
- **Fine-tuning:** 2.ª fase. Se **descongelan las últimas capas convolucionales** y se re-entrenan con un learning rate **muy pequeño** (ej. **1e-5**) para especializar los filtros en residuos sin destruir el conocimiento de ImageNet.

**Otros términos:** ImageNet · Backbone · GAP · Dense · ReLU · Softmax · Dropout · Depthwise separable · Residuales invertidos.

---

### 🔷 9 — El “Techo de Cristal” (70.88 %) — 1 min

**🎤 Qué decir:** "El resultado reveló el techo: **98 % en entrenamiento pero 70.88 % en test**."

### 📖 Qué es el sobreajuste (overfitting) — definición que pediste
> Es cuando la red **memoriza los datos de entrenamiento** (incluidos fondos e iluminación) pero **no aprende a generalizar** a imágenes nuevas. Se detecta porque:
> - **Train accuracy** sube (98 %) pero **Val accuracy** se estanca (70.88 %).
> - **Train loss** baja pero **Val loss** sube.
> La red se “aprende de memoria” en vez de entender texturas y formas. Por eso el modelo fallaba con imágenes reales.

**📖 Otros términos:** Train vs Val loss · Matriz de confusión · Data augmentation · Techo de cristal.

---

## 2.2 — Código en vivo: Cuaderno 2

> **Abre:** `momento_1_y_2_base/2_Entrenamiento_Modelo_19Clases.ipynb` (16 celdas).
> **Objetivo:** mostrar warmup/fine-tuning, las curvas y el reporte de métricas.

| Celda | Qué se ve | Qué decir |
| :--- | :--- | :--- |
| **[MD 1 / CODE 3]** | `image_dataset_from_directory` (70/15/15) | "Cargamos el dataset y hacemos la partición **estratificada** 70/15/15." |
| **[CODE 6]** | Construcción de MobileNetV2 | "Backbone `include_top=False` congelado + GAP + Dense(256) + Dropout + Softmax(19)." |
| **[CODE 7]** ⭐ | Warmup + fine-tuning | "**Etapa 1: warmup** con backbone congelado, Adam 1e-3, 20 épocas. **Etapa 2: fine-tuning** de las últimas capas con Adam **1e-5**, 30 épocas." |
| **[CODE 9]** | Curvas train/val acc y loss | "Aquí se ve el **sobreajuste**: train sube y val se estanca." |
| **[CODE 10]** ⭐ | `classification_report` | "Métricas por clase: **precision, recall, f1-score y support**." |
| **[CODE 14]** ⭐ | `make_gradcam_heatmap` | "**Grad-CAM**: mapa de calor que muestra **dónde está mirando la red**." |

### 📖 Qué es cada cosa en el código (definiciones que pediste)

**Warmup y Fine-tuning en el código**
> - **Warmup (Etapa 1):** `base_model.trainable = False` → se entrena **solo la cabeza** con `Adam(lr=1e-3)`.
> - **Fine-tuning (Etapa 2):** `base_model.trainable = True` y se **descongelan solo las últimas 35 capas** (las demás se congelan) con `Adam(lr=1e-5)` (100 veces menor).

**Train accuracy vs Val accuracy**
> - **Train accuracy:** % de acierto sobre los datos **con los que se entrenó**.
> - **Val accuracy:** % de acierto sobre el conjunto de **validación** (datos no vistos en cada actualización).
> - Si **train sube y val no**, hay sobreajuste.

**Train loss vs Val loss**
> - **Train loss:** error en entrenamiento.
> - **Val loss:** error en validación.
> - **Train loss baja + Val loss sube = sobreajuste.** Es la señal de alarma.

**`precision recall f1-score support`**
> Es el **reporte de clasificación** por clase:
> - **Precision (precisión):** de lo que predije como clase X, ¿cuánto era de verdad X? (aciertos / predicciones).
> - **Recall (exhaustividad):** de todas las imágenes de clase X, ¿cuántas detecté? (aciertos / reales).
> - **F1-score:** media armónica de precision y recall (equilibra ambos; 0 a 1, mejor cerca de 1).
> - **Support:** **nº de imágenes reales** de esa clase en el test. Sirve para saber cuánto pesa cada clase.

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
> Técnica de **explicabilidad**: usa los gradientes de la predicción respecto a la última capa convolucional para generar un **mapa de calor** sobre la imagen que muestra **qué zonas pesaron más** en la decisión. En el proyecto sirve para **auditar** si el modelo mira el residuo o el fondo. (En el código: `make_gradcam_heatmap`.)

---

## 2.3 — Glosario y conceptos clave del Cuaderno 2

| Concepto | Definición corta |
| :--- | :--- |
| **Transfer Learning** | Reutilizar pesos de ImageNet. |
| **Warmup** | Entrenar solo la cabeza con backbone congelado. |
| **Fine-tuning** | Descongelar últimas capas con lr muy bajo. |
| **Overfitting** | Memoriza y no generaliza. |
| **Train/Val accuracy** | Acierto en entrenamiento vs validación. |
| **Train/Val loss** | Error en entrenamiento vs validación. |
| **Precision / Recall / F1** | Acierto al predecir / cobertura / media armónica. |
| **Support** | Nº de imágenes reales de la clase. |
| **Grad-CAM** | Mapa de calor de dónde mira la red. |
| **BatchNorm congelado** | No actualizar sus estadísticas en fine-tuning. |

---

# 🟢 BLOQUE IV — CUADERNO 3 (Regularización)

## 3.1 — Diapositivas del Cuaderno 3 (10–11)

### 🔷 10 — Planteamiento de la Ablación — 1 min

**🎤 Qué decir:** "¿Basta con regularizar? Hicimos una **ablación controlada** —mismas semillas, mismo dataset— con Dropout, Dropout+L2 y Dropout+L1."

### 📖 Qué es cada cosa (definiciones que pediste)

**Variante A · Dropout puro**
> **Desconecta el 30 % de las neuronas en cada pasada.** Evita la **co-adaptación entre neuronas**: impide que unas neuronas dependan siempre de otras, obligando a que cada una aprenda por sí sola. Así la red se vuelve más robusta y no memoriza.

**Variante B · + L2 (Ridge)**
> **Penaliza los pesos grandes: λ·Σw².** **Contrae suavemente la magnitud de los pesos**, repartiendo la importancia entre muchas conexiones pequeñas en vez de pocas gigantes. Menos sobreajuste, pero sin apagar nada.

**Variante C · + L1 (Lasso)**
> **Fuerza dispersión (sparsity): λ·Σ|w|.** **Pone en cero conexiones espurias de fondo**, es decir, “apaga” las neuronas/conexiones que solo memorizaban ruido de fondo. Deja solo lo esencial.

**Otros términos:** Ablación · Regularización · Dispersión (sparsity) · Semilla (seed) · Brecha (gap).

---

### 🔷 11 — Resultados de la Ablación y Conclusión — 1 min

**🎤 Qué decir:** "Dropout 66.56 %, +L2 67.19 %, +L1 68.54 %. La regularización **reduce la brecha pero no rompe el 70 %**. El cuello de botella es el backbone, la pérdida y la aumentación."

**📖 Qué es cada cosa:** Brecha (gap) · Exactitud (accuracy) · Cuello de botella.

---

## 3.2 — Código en vivo: Cuaderno 3

> **Abre:** `momento_1_y_2_base/3_Defensa_Tecnica_Regularizacion.ipynb` (16 celdas).

| Celda | Qué se ve | Qué decir |
| :--- | :--- | :--- |
| **[CODE 6]** ⭐ | `construir_cabeza(reg_type)` | "La **fábrica de modelos**: `dropout_only`, `dropout_l2` (λ=1e-4) y `dropout_l1` (λ=1e-4)." |
| **[CODE 8]** | Benchmark (carga/ejecuta) | "Entrenamos las **3 variantes con la misma semilla**." |
| **[CODE 10]** | Curvas + brecha (`gap`) | "Comparación de **train vs val loss**: la brecha de generalización." |
| **[CODE 12]** ⭐ | Tabla de resultados | "**66.56 %, 67.19 % y 68.54 %**: L1 reduce la brecha pero no rompe el 70 %." |

**Monólogo de cierre (30 s):**
> "En el código ven que las 3 variantes son idénticas salvo el regularizador. Eso es una **ablación**: aislar el efecto de una sola variable. El resultado demuestra que el problema no era solo regularizar."

---

## 3.3 — Glosario y conceptos clave del Cuaderno 3

| Concepto | Definición corta |
| :--- | :--- |
| **Ablación** | Cambiar un componente a la vez. |
| **Regularización** | Técnicas anti-sobreajuste. |
| **Dropout** | Apaga un % de neuronas al azar. |
| **Co-adaptación** | Neuronas que dependen unas de otras. |
| **L2 / Ridge** | Penaliza Σw² (contrae pesos). |
| **L1 / Lasso** | Penaliza Σ|w| (vuelve cero). |
| **Sparsity (dispersión)** | Muchas conexiones en cero. |
| **Gap (brecha)** | Val loss − Train loss. |
| **Semilla** | Fija el azar para reproducir. |

---

# 🟢 BLOQUE V — CUADERNO 4 (Optimización)

## 5.1 — Diapositivas del Cuaderno 4 (12–15)

### 🔷 12 — EfficientNetV2-B0 — 1 min

**🎤 Qué decir:** "Migramos a **EfficientNetV2-B0**: bloques **Fused-MBConv** + atención **SE**. Top-1 ImageNet **78.7 %** vs 72 % de MobileNetV2. Preprocesamiento nativo [0,255]."

### 📖 Qué es cada cosa (definiciones que pediste)

**¿Por qué la nueva arquitectura EfficientNetV2-B0?**
> Porque MobileNetV2 se quedó en el **70.88 %** y el cuello de botella era la **capacidad del backbone**. EfficientNetV2-B0 ofrece **más capacidad con bajo costo** (~6 M de parámetros, 6.4 MB cuantizado), lo que permite **más precisión sin sacrificar el despliegue móvil**.

**Fused-MBConv**
> Bloque que **fusiona** las convoluciones: en lugar de hacer una convolución depthwise separable “separada” (como MBConv), usa una **convolución 3×3 estándar fusionada** con el resto del bloque. Resultado: **más rápido en CPU/GPU** y **más preciso** en texturas locales.

**ImageNet Top-1**
> Es el **porcentaje de acierto de la clase correcta como primera predicción** sobre el benchmark ImageNet (1,000 clases). EfficientNetV2-B0 = **78.7 %**; MobileNetV2 = **72.0 %**. Es la “nota” del backbone antes de adaptarlo a residuos.

**`include_preprocessing=True`**
> Permite que el modelo **integre su propia normalización** internamente: recibe **píxeles nativos [0, 255]** y él los transforma. Así se evita el error de preprocesar dos veces o con el rango equivocado.

**Otros términos:** Squeeze-and-Excitation (SE) · Backbone · Parámetros.

---

### 🔷 13 — MixUp, CutMix y Focal Loss — 1.5 min

**🎤 Qué decir:** "MixUp mezcla imágenes y etiquetas; CutMix pega un parche; Focal Loss suprime el gradiente de las fáciles."

### 📖 Qué es cada cosa (definiciones que pediste)

**MixUp (α = 0.2)**
> Es una técnica de aumentación que hace una **mezcla lineal convexa de dos imágenes** y sus etiquetas:
> $$x = \lambda x_1 + (1-\lambda) x_2 \qquad y = \lambda y_1 + (1-\lambda) y_2$$
> **Resultado:** fronteras de decisión **más suaves** (el modelo no se vuelve “rígido” entre clases). **α = 0.2** controla que la mezcla sea suave.

**CutMix (α = 1.0)**
> Recorta un **parche rectangular de una imagen** y lo **pega sobre otra**, ponderando la etiqueta por la **fracción de área** del parche. **Resultado:** evita que la red dependa de **una sola pista visual** (como el tapón o la etiqueta) para clasificar.

**Focal Loss (γ = 2.0, α = 0.25)**
> Variante de cross-entropy que **reduce el peso de los ejemplos fáciles** y **redirige el gradiente a las clases difíciles y minoritarias**:
> $$\mathcal{L}_{FL}(p_t) = -\alpha (1 - p_t)^{\gamma} \log(p_t)$$
> - **γ (gamma) = 2.0:** controla cuánto se suprime lo fácil.
> - **α (alpha) = 0.25:** pondera el balance entre clases.

**El ejemplo numérico de la diapositiva:**
> Si **p_t = 0.95** (el modelo está muy seguro y acierta), el factor modulador es:
> $$(1 - 0.95)^2 = 0.0025$$
> Es decir, **suprime el gradiente un 99.75 %**. La red deja de “perder el tiempo” en lo fácil y se enfoca en **pilas, aceites y sanitarios** (las clases difíciles).

**Otros términos:** Gradiente · Clase minoritaria · Cross-entropy · p_t.

---

### 🔷 14 — Poda de Ruido y Ensamble Calibrado — 1.5 min

**🎤 Qué decir:** "Confident Learning poda muestras con p < 0.12. Ensamble 80 % Focal + 20 % Base, con TTA y Temperature Scaling."

### 📖 Qué es cada cosa (definiciones que pediste)

**Confident Learning (Cleanlab / MIT)**
> Método que **detecta y poda muestras con etiqueta ruidosa** comparando la predicción del modelo con la etiqueta original. Si la probabilidad de la etiqueta es muy baja (**p_etiqueta < 0.12**), se marca como sospechosa.
> - **133 muestras fuera de dominio depuradas (7.8 % del test):** estanterías, diagramas médicos, capturas con texto que se colaron en el scraping.

**Ensamble homogéneo: P = 0.80·Focal + 0.20·Base**
> Es la **fusión ponderada** de las predicciones de dos modelos: el **Focal** (especialista en clases difíciles) pesa **80 %**, y el **Base** (más calibrado en clases mayoritarias) pesa **20 %**.

**Multi-Angle TTA: 0.50·orig + 0.30·flipH + 0.20·flipV**
> **Test-Time Augmentation:** en inferencia se evalúa la imagen en **3 ángulos** (original, volteo horizontal, volteo vertical) y se **promedian** las predicciones con esos pesos, cancelando el sesgo de orientación del residuo.

**Temperature Scaling (T = 0.75)**
> **Calibración** que **divide los logits entre T (0.75)** antes del Softmax:
> $$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
> **Resultado:** **agudiza la probabilidad de la clase correcta**, corrigiendo la sobreconfianza difusa.

**Otros términos:** Ruido de etiqueta · Ensamble · Calibración · Logits.

---

## 5.2 — Código en vivo: Cuaderno 4

> **Abre:** `momento_1_y_2_base/4_Optimizacion_Avanzada_EfficientNetV2_90Acc_EJECUTADO.ipynb` (45 celdas).

| Celda | Qué se ve | Qué decir |
| :--- | :--- | :--- |
| **[MD 7 / CODE 8]** ⭐ | `compute_class_weight("balanced")` | "**Pesos de clase**: `w_c = N / (K · n_c)` — se compensa el desbalance." |
| **[MD 9 / CODE 10]** ⭐ | `aplicar_mixup`, `aplicar_cutmix` | "**MixUp** con α=0.2 y **CutMix** con α=1.0 usando una **Beta(α, α)**." |
| **[MD 11 / CODE 12]** | `EfficientNetV2B0(...)` | "Backbone EfficientNetV2-B0 con `include_preprocessing=True`." |
| **[CODE 14 / CODE 15]** ⭐ | Warmup + fine-tuning + BatchNorm | "**Congelamos BatchNorm** en fine-tuning para no destruir las estadísticas." |
| **[MD 31 / CODE 32]** | Confident Learning | "Poda de muestras con **p_etiqueta < 0.12**." |
| **[MD 34 / CODE 35]** | `CategoricalFocalCrossentropy` | "**Focal Loss** con γ=2.0 y α=0.25." |
| **[MD 39 / CODE 41]** ⭐ | Ensamble + TTA + Temperature | "**0.80 Focal + 0.20 Base**, TTA y T=0.75." |

### 📖 Qué es cada cosa en el código (definiciones que pediste)

**Sección 4 — ¿Qué son los pesos de clase y por qué se calculan?**
> El dataset está **desbalanceado**: hay clases con muchas fotos y otras con pocas. Si no se corrige, la red tiende a **predecir las clases mayoritarias** y a ignorar las minoritarias.
>
> Se calculan con la formulación balanceada estándar de **He et al.**:
> $$w_c = \frac{N}{K \cdot n_c}$$
> - **N** = total de muestras de entrenamiento.
> - **K** = número de clases (**19**).
> - **n_c** = número de muestras de la clase **c**.
>
> **Por qué se hace:** con esta fórmula, una clase con **pocas** muestras recibe un peso **mayor** (se refuerza en la pérdida) y una clase con **muchas** recibe un peso **menor**. Así **la pérdida no ignora las clases minoritarias** y el modelo aprende de todas por igual.

**Sección 5 — Aumento de datos (fotométrico y geométrico), MixUp y CutMix**
> Para **erradicar el sobreajuste del 98 % en entrenamiento** y enseñar al modelo **invariancia espacial y de iluminación**, se aplica:
> 1. **Aumento fotométrico y geométrico:** rotaciones aleatorias de **±15 %**, variaciones de **zoom**, **traslación**, y variaciones de **brillo y contraste**. (En el código: `RandomFlip`, `RandomRotation(0.15)`, `RandomZoom`, `RandomTranslation`, `RandomBrightness(0.18)`, `RandomContrast(0.18)`.)
> 2. **MixUp (Zhang et al., 2018):** mezcla lineal convexa de pares de imágenes (`x = λx₁ + (1−λ)x₂`) y sus etiquetas (`y = λy₁ + (1−λ)y₂`). Obliga a la red a un comportamiento **lineal y continuo** en el espacio de representación.
> 3. **CutMix (Yun et al., 2019):** recorte rectangular de un objeto pegado sobre otro residuo, **ponderando la etiqueta por la fracción de área**. Evita que la red dependa de **una única pista visual** (una etiqueta o un tapón) para clasificar.

**Otros términos del código:** `lambda` (λ) · distribución Beta · `class_weight` · BatchNorm congelado · TTA · Temperature Scaling.

---

## 5.3 — Glosario y conceptos clave del Cuaderno 4

| Concepto | Definición corta |
| :--- | :--- |
| **EfficientNetV2-B0** | Backbone más capaz y eficiente que MobileNetV2. |
| **Fused-MBConv** | Bloque que fusiona convoluciones (más rápido/preciso). |
| **Squeeze-and-Excitation** | Atención que repondera canales. |
| **ImageNet Top-1** | Acierto de la clase correcta en ImageNet. |
| **include_preprocessing** | Normalización interna del modelo (píxeles nativos). |
| **MixUp (α=0.2)** | Mezcla convexa de imágenes y etiquetas. |
| **CutMix (α=1.0)** | Pega un parche de otra imagen. |
| **Focal Loss (γ=2)** | Reduce el peso de ejemplos fáciles. |
| **Pesos de clase** | `w_c = N/(K·n_c)` para compensar desbalance. |
| **Confident Learning** | Poda de etiquetas ruidosas. |
| **Ensamble** | Combinar 0.80 Focal + 0.20 Base. |
| **TTA** | Promediar predicciones de volteos. |
| **Temperature Scaling** | Calibrar confianza (T=0.75). |

---

# 🟢 BLOQUE VI — DESPLIEGUE Y CIERRE (Diapositivas 16–18)

## 🔷 16 — TensorFlow Lite — 1 min
**🎤 Qué decir:** "Cuantización INT8: de **113 MB a 6.4 MB**, **15–25 ms**. App Android **100 % offline**."
**📖 Qué es cada cosa:** Cuantización · INT8 / Dynamic Range · TFLite (LiteRT) · Latencia · SLA (< 2,000 ms) · APK.

## 🔷 17 — Demo Web — 1 min
**🎤 Qué decir:** "Funcionando en vivo: **bombilla 97 %**, **escombros 98.2 %**, **sanitarios 99.4 %**. Muestra Top de probabilidades y recomendación normativa."
**📖 Qué es cada cosa:** Inferencia · Top de probabilidades · Categoría normativa · Recomendación.
> ⚠️ **No demuestres con vidrio, botellas, plásticos ni aceites:** ahí están las confusiones ópticas ya explicadas.

## 🔷 18 — Conclusiones y ODS — 1 min
**🎤 Qué decir:** "Del cuaderno al campo: **70.88 % → 92.49 %**; Edge AI viable (6.4 MB, 20 ms); impacto en **ODS 11, 12, 3 y 13**."
**📖 Qué es cada cosa:** ODS 11 · 12 · 3 · 13 · Economía circular.

---

# ⚠️ ALERTAS DE COHERENCIA (diapositiva ↔ código)

Antes de exponer, **revisa estas 3 diferencias** para que no te contradiga el jurado al pasar de diapositiva a código:

| Tema | La diapositiva dice | El notebook ejecuta | Recomendación |
| :--- | :--- | :--- | :--- |
| **Dropout (C2/C3)** | "Dropout(0.3)" y "30 %" | `Dropout(0.4)` en notebook 2 y 3 | Di "**Dropout**" sin el número exacto, o aclara "en el código usé **40 %**". |
| **Fine-tuning (C2)** | "últimas 30 capas, lr=1e-4" | **35 capas, lr=1e-5**, 30 épocas | Al mostrar el código, di los números del código. |
| **Warmup (C2)** | (no dice épocas) | **20 épocas** de warmup | Di "20 épocas de warmup, 30 de fine-tuning". |

> 💡 Si quieres, puedo **alinear las diapositivas a los números del código** (1 minuto de trabajo) para que no haya ninguna diferencia. Dime y lo hago.

---

# ✅ CHECKLIST ANTES DE EXPONER

- [ ] Abrir la presentación (18 diapositivas) y verificar que cargan las imágenes.
- [ ] Abrir los **4 notebooks** en orden (1→2→3→4) y dejar las celdas clave listas.
- [ ] Tener el aplicativo web en `http://127.0.0.1:7865` si harás demo en vivo.
- [ ] Memorizar: **92.49 % · 98.24 % · 0.921 · 6.4 MB · 20 ms**.
- [ ] Repasar las definiciones que pediste: **pHash/Hamming · Transfer Learning · Overfitting · Grad-CAM · Dropout/L1/L2 · Fused-MBConv · MixUp/CutMix/Focal Loss · Pesos de clase · Confident Learning · TTA · Temperature Scaling**.
- [ ] Tener listas las 5 respuestas del banco de preguntas.

---

> 📌 **Documentos vinculados:**
> - Diapositivas: [`Exposicion_Final_EcoClasificador_92acc.pptx`](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/Exposicion_Final_EcoClasificador_92acc.pptx)
> - Guión de estudio con glosario: [`GUION_ARTICULADO_CON_GLOSARIO.md`](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/GUION_ARTICULADO_CON_GLOSARIO.md)
> - Glosario: [`glosario.md`](file:///home/jhank/Desktop/ModeloCNN/glosario.md)
> - Notebooks: `momento_1_y_2_base/1_…`, `2_…`, `3_…`, `4_…_EJECUTADO.ipynb`
