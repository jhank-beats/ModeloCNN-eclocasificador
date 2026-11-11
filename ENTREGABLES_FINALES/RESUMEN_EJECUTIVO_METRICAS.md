# Resumen Ejecutivo de Métricas y Hallazgos
## Proyecto Clasificador de Residuos (19 Clases) - Bienal UCC 2026

### 1. Ficha Técnica del Dataset
* **Imágenes Totales:** 11,541 fotos normalizadas a 180×180 px RGB.
* **Número de Clases:** 19 categorías.
* **Clases Propias de la Región (10):** 6,789 fotos (58.8% del dataset) con más de 300 a 500 fotos por clase.
* **Clases Base RealWaste (9):** 4,752 fotos (41.2%).
* **División:** 70% Entrenamiento (8,081), 15% Validación (1,726), 15% Prueba Independiente (1,734).

### 2. Ficha Técnica del Modelo
* **Backbone:** MobileNetV2 (Preentrenado en ImageNet-1k).
* **Parámetros Totales:** 2,514,259 parámetros.
* **Entrenamiento:** 2 Fases (Feature Extraction con Adam $lr=10^{-3}$ + Fine-Tuning quirúrgico con $lr=10^{-5}$ y BatchNorm congelado).
* **Optimización de Hiperparámetros:** Keras Tuner (10 ensayos). Mejor configuración: Dense Units = 256, Dropout = 0.3, Learning Rate = $10^{-3}$.

### 3. Rendimiento y Benchmark de Despliegue (Edge AI)
* **Formato Móvil:** TensorFlow Lite (TFLite) con Cuantización de Rango Dinámico.
* **Peso en Disco:** **2.7 MB** (frente a los 9.2 MB del modelo original, reducción del 71%).
* **Consumo de Memoria RAM en Smartphone:** ~17.5 MB.
* **Latencia en Teléfonos Reales:**
  * Xiaomi Redmi Note 11 (Snapdragon 680): **14.6 ms/imagen** (~68 FPS).
  * Samsung Galaxy A13 (Exynos 850): **24.1 ms/imagen** (~41 FPS).
  * **Cumplimiento del SLA (< 2,000 ms):** Superado por más de 100 veces.
* **Accuracy en Prueba Independiente:** **88.1%**.
* **Prueba de Campo con 20 Fotos Reales de la Región:** **18 aciertos de 20 (90.0% de efectividad en campo)**.
