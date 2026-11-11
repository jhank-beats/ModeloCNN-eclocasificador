# GLOSARIO TÉCNICO DE ESTUDIO — MACHINE LEARNING Y DEEP LEARNING
## Conceptos clave, arquitecturas CNN, entrenamiento y despliegue móvil
### Proyecto: EcoClasificador UCC (19 clases de residuos)

> **Cómo usar este documento:** léelo completo una vez. Para la defensa, memoriza la **sección 6 (guiones)** y repasa la **sección 7 (aclaraciones)**. Las definiciones de las secciones 1–5 son las que usas si el jurado pregunta "¿y eso qué es?".

> ⚠️ **Nota de actualización:** este glosario se alineó primero a la versión **MobileNetV2** (TFLite de 2.89 MB, 14.5 ms). La versión final usa **EfficientNetV2** → **6.4 MB** y **15–25 ms**. Donde el texto diga 2.89 MB, di **6.4 MB**.

---

## 1. Arquitecturas CNN y bloques de visión artificial

| Término / Sigla | Definición clara y aplicación en el proyecto |
| :--- | :--- |
| **Backbone** | Red neuronal base (tronco extractor de características visuales como bordes, texturas y formas) sobre la cual se conecta el cabezal de clasificación final. |
| **MobileNet / MobileNetV2** | Familia de redes ligeras diseñadas para celulares. MobileNetV2 utiliza convoluciones separables en profundidad y bloques de residuos invertidos, logrando alta precisión con solo ~2.5 M de parámetros (3.4 M con la cabeza) y bajo consumo de batería. |
| **ResNet50** | Arquitectura profunda de 50 capas basada en conexiones residuales (*skip connections*) que evitan el desvanecimiento del gradiente. En el proyecto se comparó contra MobileNetV2 como arquitectura pesada. |
| **YOLO (You Only Look Once)** | Modelo de visión artificial en tiempo real que divide la imagen en una rejilla para detectar o clasificar objetos en una sola pasada. |
| **Convolución Estándar 2D** | Operación clásica que desliza un filtro (*kernel*) sobre alto, ancho y todos los canales simultáneamente. Es potente pero computacionalmente costosa. |
| **Depthwise Separable Convolutions** | Convolución separable en profundidad (usada en MobileNetV2): divide la operación en dos pasos (1.º un filtro ligero por cada canal espacial y 2.º una combinación 1×1 o *pointwise*), reduciendo el costo computacional hasta 8–9 veces. |
| **Bottleneck (cuello de botella)** | Bloque que comprime o expande el número de canales con convoluciones 1×1 para reducir el número de operaciones sin perder capacidad de representación. |
| **ImageNet** | Base de datos masiva (>1.2 millones de imágenes y 1,000 categorías) con la que fueron pre-entrenados los modelos base (MobileNetV2 y ResNet50) antes de adaptarlos a los 19 residuos. |
| **EfficientNetV2-B0** | CNN más moderna y eficiente (Google): usa bloques *Fused-MBConv* + atención SE. Versión B0 ≈ 5.9 M de parámetros y Top-1 ImageNet **78.7 %** (vs 72 % de MobileNetV2). Es el backbone del Cuaderno 4. |
| **Fused-MBConv** | Bloque que fusiona convoluciones (reemplaza la convolución separable por una 3×3 estándar fusionada): más rápido en CPU/GPU y más preciso en texturas locales. |
| **Squeeze-and-Excitation (SE)** | Módulo de atención que "aprieta" (pool global) y "excita" (repondera) los canales según su importancia. |

---

## 2. Capas de la red y fundamentos matemáticos

| Término / Sigla | Definición clara y aplicación en el proyecto |
| :--- | :--- |
| **Tensor** | Arreglo multidimensional de números (matriz N-dimensional). Por ejemplo, una imagen de entrada al modelo es un tensor de forma (180, 180, 3). |
| **Weights (pesos)** | Parámetros numéricos internos que la red aprende durante el entrenamiento y que determinan la importancia de cada conexión neuronal. |
| **GAP (GlobalAveragePooling2D)** | Capa que calcula el promedio espacial de cada mapa de características (ej. convierte un bloque 6×6×1280 en un vector plano de 1280 valores), reduciendo drásticamente el número de parámetros frente a *Flatten*. |
| **Dense / Dense Units** | Capa totalmente conectada (*Fully Connected*) donde cada neurona recibe señales de todas las neuronas de la capa anterior. Se usa en el cabezal clasificador (ej. 256 unidades). |
| **ReLU (Rectified Linear Unit)** | Función de activación no lineal definida como f(x) = max(0, x). Convierte valores negativos en cero y permite aprender patrones complejos rápidamente. |
| **Softmax** | Función de activación en la última capa (19 neuronas) que transforma los puntajes brutos (*logits*) en una distribución de probabilidades que suman exactamente 1.0 (100 %). |
| **BatchNorm (BatchNormalization)** | Capa que normaliza las activaciones internas de cada lote (media 0 y varianza 1), acelerando la convergencia y estabilizando el entrenamiento. |

---

## 3. Entrenamiento, regularización y optimización

| Término / Sigla | Definición clara y aplicación en el proyecto |
| :--- | :--- |
| **Transfer Learning** | Aprendizaje por transferencia: reutilizar el conocimiento visual aprendido en ImageNet como punto de partida en lugar de entrenar la red desde cero. |
| **Fine-Tuning (ajuste fino)** | Segunda fase del entrenamiento donde se descongelan las últimas capas del backbone y se re-entrenan con una tasa de aprendizaje muy pequeña (ej. 1e-5) para especializarlas en las 19 clases de residuos. |
| **Regla de oro de ingeniería** | ⚠️ **¡Ojo en la defensa!** Tiene dos aplicaciones clave: **(1) Congelar BatchNormalization**: al hacer fine-tuning, las capas BatchNorm deben mantenerse en modo inferencia (congeladas) para no destruir las estadísticas de media y varianza aprendidas en ImageNet. **(2) Contrato de preprocesamiento**: escalar los píxeles al rango correcto del backbone — `[-1, 1]` para MobileNetV2; `[0, 255]` con `include_preprocessing=True` para EfficientNetV2. |
| **Dropout** | Técnica de regularización que apaga aleatoriamente un porcentaje de neuronas (ej. 30 % o 40 %) en cada paso del entrenamiento para evitar que el modelo memorice (*overfitting*). |
| **Dropout + L2 / L1** | Combinación de Dropout con penalización sobre la magnitud de los pesos (L2 = penalización cuadrática / Ridge; L1 = penalización absoluta / Lasso) en la función de pérdida para forzar pesos pequeños y mejorar la generalización. Ver explicación ampliada en **3.1**. |
| **Adam** | Optimizador adaptativo que ajusta automáticamente la tasa de aprendizaje de cada parámetro combinando *Momentum* y *RMSProp*. |
| **AdamW** | Variante de Adam con **decaimiento de peso** (*weight decay*) desacoplado: penaliza pesos grandes dentro del propio optimizador. En el proyecto: `AdamW(lr=1e-4, weight_decay=1e-4)`. |
| **Label Smoothing** | Suaviza las etiquetas: en vez de "100 % esta clase", usa "~90 % esta clase y el resto repartido". Evita la sobreconfianza. Ej.: con 19 clases y ε=0.1, la clase correcta vale ~0.905 y las demás ~0.005. |
| **Cross-Entropy Loss** | Entropía cruzada categórica: función de pérdida estándar en clasificación multiclase que penaliza fuertemente las predicciones seguras pero equivocadas. |
| **Focal Loss** | Variante de Cross-Entropy que multiplica por `(1−p_t)^γ` para **reducir el peso de los ejemplos fáciles** y concentrarse en los difíciles. En el proyecto: `γ=2.0`, `α=0.25`. |
| **EarlyStopping** | Callback que vigila la pérdida de validación (`val_loss`) y detiene el entrenamiento automáticamente cuando deja de mejorar tras varias épocas (*patience*), restaurando los mejores pesos. |
| **ReduceLROnPlateau** | Callback que reduce el *Learning Rate* (ej. a la mitad) cuando el entrenamiento se estanca en una meseta, permitiendo un ajuste más fino. |
| **CosineDecay** | Planificador que baja el learning rate siguiendo una curva cosenoidal (de `1e-4` a `1e-6`), con un warmup suave inicial. |
| **Random Search / Hyperband** | Algoritmos de búsqueda automática de hiperparámetros. Random Search prueba combinaciones al azar; Hyperband entrena muchas configuraciones por pocas épocas y asigna más recursos solo a las más prometedoras. |

### 3.1 Dropout + L2 vs Dropout + L1 (explicación ampliada)

> **La frase de defensa (memorizar):**
> *"**Dropout + L2** es apagar neuronas al azar más una penalización cuadrática de los pesos: los contrae sin apagarlos. **Dropout + L1** es apagar neuronas más una penalización absoluta que lleva muchos pesos a cero, eliminando conexiones que solo memorizaban el fondo. **L2 'baja el volumen', L1 'silencia'**."*

| | **Dropout + L2 (Ridge)** | **Dropout + L1 (Lasso)** |
| :--- | :--- | :--- |
| **Fórmula de pérdida** | `L = L_CE + λ·Σw²` | `L = L_CE + λ·Σ|w|` |
| **Qué penaliza** | El cuadrado del peso | El valor absoluto del peso |
| **Efecto** | **Contrae** todos los pesos, pero ninguno llega a 0 | **Lleva muchos pesos a 0** (dispersión / *sparsity*) |
| **Analogía** | "Baja el volumen a todos, pero todos siguen hablando" | "Silencia del todo a los que no aportan" |
| **En el Cuaderno 3** | Variante B → **67.19 %** (brecha 1.1015) | Variante C → **68.54 %** (brecha 0.9969, la menor) |

**Por qué se combinan con Dropout:** actúan en frentes distintos — el Dropout regulariza las **activaciones** (forward pass) y L1/L2 regularizan los **pesos** (dentro de la pérdida). Juntos atacan el sobreajuste por dos lados.

---

## 4. Métricas, formatos y despliegue en móviles (Edge AI)

| Término / Sigla | Definición clara y aplicación en el proyecto |
| :--- | :--- |
| **Train Loss / Val Loss** | *Train Loss*: error del modelo en los datos de entrenamiento. *Val Loss*: error en el conjunto de validación (datos no vistos al actualizar pesos). Si Train Loss baja pero Val Loss sube, hay sobreajuste (*overfitting*). |
| **Val (Validation Set)** | Conjunto de validación (15 % del dataset en el proyecto) utilizado al final de cada época para ajustar hiperparámetros y activar EarlyStopping. |
| **Acc (Accuracy / exactitud)** | Porcentaje total de predicciones correctas sobre el total de imágenes evaluadas. |
| **Top-1 / Top-2 / Top-3** | Acierto si la clase correcta está entre las k predicciones más probables. En el proyecto: **92.49 % / 98.24 % / 99.80 %**. |
| **F1-Score** | Métrica que combina precisión y recall. Ver explicación ampliada en **4.1**. |
| **GFLOPs** | Miles de millones de operaciones de punto flotante por segundo; mide el peso computacional matemático de una red neuronal. |
| **.keras** | Formato moderno oficial de TensorFlow/Keras (archivo comprimido con arquitectura, pesos y configuración) usado para guardar el modelo maestro. |
| **TFLite (TensorFlow Lite / LiteRT)** | Formato ultraligero (`.tflite`) optimizado mediante cuantización para ejecutar inferencia local en Android sin internet. En la versión final: **6.4 MB**. |
| **SLA (Service Level Agreement)** | Acuerdo de Nivel de Servicio (¡no "Statistical Learning Algorithm"!). En el proyecto es el requisito del docente: inferencia móvil **< 2,000 ms** por foto. El modelo logra **15–25 ms**, superándolo con creces. |

### 4.1 ¿Qué es el F1-Score? (explicación ampliada)

> **Definición base:** *"El **F1-Score** (o puntuación F1) es una métrica de evaluación utilizada en aprendizaje automático (*machine learning*) y estadística para medir el rendimiento de un modelo de clasificación."*

**Cómo se calcula:**
```
F1 = 2 · (Precisión · Recall) / (Precisión + Recall)
```
Es la **media armónica** de Precisión y Recall.

| Concepto | Qué mide | Fórmula simple |
| :--- | :--- | :--- |
| **Precisión** | De lo que predije como clase X, ¿cuánto era realmente X? | aciertos / predicciones |
| **Recall** | De todas las imágenes reales de X, ¿cuántas detecté? | aciertos / reales |
| **F1** | Equilibrio entre ambos | media armónica |

**¿Por qué media armónica y no promedio?** Porque **castiga** cuando una de las dos es baja: si Precisión = 1.0 y Recall = 0.1, el F1 es ~0.18 (no 0.55). Obliga a que el modelo sea bueno en **ambas**.

**F1 Macro (el del proyecto):** se calcula el F1 de **cada clase por separado** y luego se promedian **con el mismo peso**. Así una clase con muchas imágenes no "tapa" a una minoritaria. En el proyecto: **F1 Macro = 0.921**, lo que certifica que el modelo es equilibrado entre las 19 clases.

---

## 5. CRISP-ML(Q) y hashes MD5

### ¿Qué es CRISP-ML(Q)?
> **Cross-Industry Standard Process for the Development of Machine Learning Applications with Quality Assurance.** Es la metodología estándar de ingeniería para proyectos de ML con aseguramiento de calidad, en **6 fases**:
>
> 1. **Comprensión del negocio y de los datos:** definir las 19 categorías y recolectar las imágenes (RealWaste + dataset suplementario).
> 2. **Ingeniería de datos:** limpieza con hashes, partición estratificada 70/15/15 y data augmentation.
> 3. **Modelado:** Transfer Learning, comparativa de arquitecturas y entrenamiento en 2 fases.
> 4. **Evaluación de calidad:** test independiente, matriz de confusión y F1-Score.
> 5. **Despliegue (Edge AI):** cuantización a TensorFlow Lite e integración en la app móvil.
> 6. **Monitoreo y pruebas de campo:** validación en teléfonos reales verificando el SLA (< 2,000 ms).

### ¿Qué es un Hash / MD5?
> Es una **función matemática que genera una "huella digital" alfanumérica única de 128 bits** a partir del contenido binario de un archivo. Si dos imágenes tienen exactamente los mismos bytes (aunque tengan distinto nombre), su hash MD5 es idéntico.

**¿Para qué se usa en el proyecto?**
1. **Eliminación de duplicados / integridad del dataset:** al integrar imágenes desde la web, se calcula el MD5 de cada foto para detectar y eliminar repetidas, evitando que una misma foto caiga en entrenamiento y en prueba a la vez (*data leakage*).
2. **Verificación de integridad del modelo** (`.keras` / `.tflite`): garantizar que el archivo exportado a la app no se corrompió al transferirse.

**dHash vs MD5 (no confundir):** el **dHash (perceptual)** detecta imágenes *visualmente* parecidas (aunque cambien de compresión); el **MD5 (criptográfico)** detecta copias *byte-idénticas* y contradicciones entre carpetas. Son **capas complementarias**.

---

## 6. Guiones de defensa (para memorizar)

### 6.1 Entrenamiento en dos fases (Warmup + Fine-tuning)

> *"Entrenamos en dos fases. En el **warmup** congelamos el backbone y entrenamos solo la cabeza a **1e-3**, con **Label Smoothing 0.1** para que el modelo no sea sobreconfiado. Después, en el **fine-tuning**, descongelamos el backbone pero dejamos **BatchNorm congelado** para no dañar las estadísticas de ImageNet, y bajamos el learning rate con **CosineDecay de 1e-4 a 1e-6**. Así adaptamos el modelo sin destruir lo que ya sabía."*

### 6.2 Parte II: del ~82 % al 92.49 %

> *"En la Parte II hicimos cuatro cosas. **Una:** limpiamos el ruido de etiquetas con **Confident Learning**: si el modelo daba menos de 5 % a la etiqueta real, esa imagen se excluía —podamos **133 muestras, el 7.8 %**. **Dos:** entrenamos con **Focal Loss (γ=2)** para que el gradiente dejara de gastarse en lo fácil y se centrara en pilas, aceites y sanitarios. **Tres:** implementamos un módulo de **destilación** con profesor **ConvNeXt-Tiny** y estudiante **EfficientNetV2-B0**. **Cuatro:** el salto final lo dio el **ensamble 80/20** con **TTA** y **Temperature Scaling**, más una poda final a **0.12**. Así llegamos a **92.49 %**."*

---

## 7. Aclaraciones rápidas para la defensa (coherencia)

| Tema | Qué decir |
| :--- | :--- |
| **Destilación** | La clase `ModeloDestilador` está **implementada** (T=3.0, α=0.4) y se **propuso** ConvNeXt-Tiny como profesor; **no se ejecutó**. El 92.49 % vino del ensamble. |
| **MobileNetV2: 2.5 M o 3.4 M** | ~2.5 M es el **backbone**; 3.4 M es el **total con la cabeza**. |
| **Umbrales de poda** | Sección 14 usa **0.05**; la poda final del ensamble usa **0.12**. |
| **TFLite: 2.89 MB o 6.4 MB** | 2.89 MB era la versión MobileNetV2; la final (EfficientNetV2) es **6.4 MB**. |
| **Fine-tuning: 30 o 35 capas** | La diapositiva dice 30 capas; el notebook 2 ejecuta las últimas **35 capas** con `lr=1e-5`. |
| **Dropout: 30 % o 40 %** | La diapositiva dice 30 %; el notebook usa `Dropout(0.4)` (40 %). |

---

> 📌 **Proyecto:** EcoClasificador Móvil con Visión Artificial y Edge AI · Universidad Cooperativa de Colombia · Electiva III.
