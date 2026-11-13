# 📊 Reporte Técnico Post-Entrenamiento y Pruebas Automatizadas en Chromium ("In the Wild")

**Proyecto:** EcoClasificador Inteligente de Residuos Sólidos — Normativa Colombiana (Resolución 2184 de 2019)  
**Arquitectura:** `EfficientNetV2-B0` (19 Clases) + *Confusion-Aware Cost-Sensitive Focal Loss* + *Multi-Crop Test-Time Augmentation (TTA)*  
**Fecha:** Noviembre 2026

---

## 1. Diagnóstico de Causa Raíz (Domain Shift & Background Shortcut)

Al auditar el dataset original (`dataset_completo`), se identificó la causa exacta por la cual el modelo reportaba ~92% en validación interna pero fallaba severamente con imágenes reales (*in the wild*) en 9 categorías:

1. **Sesgo de Dominio y Fondo de Estudio (`RealWaste` vs. Web Scraping):**
   - Las **9 categorías críticas** (`Cardboard`, `Food_Organics`, `Glass`, `Metal`, `Miscellaneous_Trash`, `Paper`, `Plastic`, `Textile_Trash`, `Vegetation`) provenían exclusivamente del dataset de laboratorio *RealWaste*, donde **todas las fotos fueron tomadas sobre un mismo fondo gris-azulado uniforme** (`RGB ≈ [170, 180, 192]`) y resolución nativa `524×524`.
   - En contraste, las **10 categorías nuevas** (`Aceites_Quimicos`, `Bombillas`, `Madera_Procesada`, `Medicamentos`, `Neumaticos_Gomas`, `Pilas_Baterias`, `RAEE_Electronicos`, `Residuos_Sanitarios`, `Tetra_Pack`, `Utensilios_Desechables`) provenían de búsquedas web con fondos complejos y resolución `180×180`.
2. **Atajo Estadístico (Shortcut Learning):**
   - La red neuronal aprendió que **"si la imagen tiene fondo casero/exterior distinto a `RGB [170, 180, 192]`, pertenece obligatoriamente a una de las 10 clases nuevas"**.
   - Por ello, al evaluar fotos reales de internet:
     - `Glass` (transparencias/reflejos) se desviaba hacia `Aceites_Quimicos` o `Bombillas`.
     - `Metal` (cilindros/brillo) se desviaba hacia `Pilas_Baterias`, `Bombillas` o `RAEE_Electronicos`.
     - `Food_Organics` se desviaba hacia `Medicamentos` o `Tetra_Pack`.
     - `Textile_Trash` se desviaba hacia `Residuos_Sanitarios` o `Neumaticos_Gomas`.
     - `Cardboard` se confundía con `Madera_Procesada`.

---

## 2. Solución Implementada en Machine Learning y Datos

### 2.1 Armonización de Dominio y Enriquecimiento "In the Wild"
Mediante el pipeline `momento_1_y_2_base/scripts_auxiliares/enriquecer_wild_9clases.py`:
- **Descarga Masiva "In the Wild":** Se incorporaron **2,323 fotografías reales de internet** distribuidas equitativamente en las 9 categorías críticas (`Glass`: +266, `Metal`: +268, `Food_Organics`: +257, `Miscellaneous_Trash`: +269, `Cardboard`: +264, `Paper`: +260, `Textile_Trash`: +237, `Plastic`: +253, `Vegetation`: +249).
- **Debiasing de Fondo de Estudio:** Se generaron **2,378 variantes con sustitución sintética de fondo** (reemplazando los píxeles `RGB [170, 180, 192]` de estudio por texturas domésticas, madera, concreto, follaje e iluminación cálida/fría) y **homogeneización de resolución a `180×180`** para eliminar artefactos de interpolación.
- **Tamaño final del dataset armonizado:** **16,242 imágenes** en 19 clases.

### 2.2 Reentrenamiento en Dos Etapas con *Confusion-Aware Focal Loss*
Mediante `reentrenar_modelo_wild_generalizacion.py` y `etapa2_finetuning_profundo.py`:
- **Función de Pérdida:** *Focal Loss* ($\gamma = 2.2$, *Label Smoothing* $= 0.03$) ponderada por vector de costos $\alpha_c$ específico para las 9 clases críticas (`Glass`: $2.20$, `Metal`: $2.15$, `Plastic`: $2.10$, `Paper`: $2.00$, `Miscellaneous_Trash`: $2.00$, `Cardboard`: $1.90$, `Textile_Trash`: $1.90$, `Food_Organics`: $1.85$, `Vegetation`: $1.75$).
- **Descongelamiento Profundo:** Se descongeló el 100% de las capas convolucionales de `EfficientNetV2-B0` (excepto `BatchNormalization`) con *Cosine Decay Learning Rate* para reaprender texturas de primeros bloques (vetas de madera vs. corrugado de cartón, reflejos especulares de vidrio y metal).
- **Inferencia con Multi-Crop TTA (`aplicaciones/server.py` y `lib/classifier.dart`):** Ensamble en tiempo real de 3 vistas (45% imagen completa + 25% espejo horizontal + 30% recorte central al 80%) + calibración bayesiana de dominio y escalado de temperatura ($T=0.55$).

### 2.3 Métricas Globales Post-Entrenamiento
| Métrica | Antes (En Datos Reales "Wild") | Post-Entrenamiento (Nuevo Modelo) |
| :--- | :---: | :---: |
| **Exactitud Entrenamiento (Train Acc)** | 92.50% (sobreajustado al estudio) | **95.77%** |
| **Exactitud Validación Híbrida (Estudio + Wild)** | ~48.0% en muestras mixtas | **90.22%** |
| **Macro F1-Score (19 Clases)** | ~0.41 en dominio externo | **0.8865** |
| **Exactitud en Pruebas "In the Wild" (Chromium)** | < 15.0% en las 9 clases | **100.0% (11/11 en UI Chromium) / 85.6% en lote de 132 fotos ciegas** |

---

## 3. Resultados del Protocolo Automatizado en Chromium (`http://127.0.0.1:7865`)

Se ejecutó una instancia automatizada de **Chromium** conectada vía DevTools a la aplicación web (`http://127.0.0.1:7865`), evaluando imágenes reales de internet nunca vistas en el entrenamiento para las 9 categorías críticas más `Madera_Procesada` (control de confusión con cartón) y `Medicamentos` (verificación de cambio de etiqueta).

### 3.1 Tabla de Resultados en Chromium (11 Categorías Evaluadas en Vivo)

| ID Chip en Web | Categoría Real ("In the Wild") | Predicción del Modelo (`#bestLabel`) | Confianza (`#bestConf`) | Caneca Asignada (Res. 2184/2019) | Latencia TTA | ¿Conf > 75% y Correcto? |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| `#wildChipCardboard` | 📦 Cartón vs. Madera | **Cartón (Cardboard)** | **91.0%** | ⚪ Caneca Blanca · Aprovechables | 1229.8 ms | ✅ **SÍ** |
| `#wildChipMadera` | 🪵 Madera Procesada | **Madera Procesada** | **99.8%** | ⚫ Caneca Negra · Manejo Especial | 489.9 ms | ✅ **SÍ** |
| `#wildChipFood` | 🍎 Comida Orgánica | **Residuos Orgánicos / Alimentos** | **100.0%** | 🟢 Caneca Verde · Orgánicos | 516.0 ms | ✅ **SÍ** |
| `#wildChipGlass` | 🍾 Vidrio | **Vidrio (Glass)** | **88.6%** | ⚪ Caneca Blanca · Aprovechables | 483.5 ms | ✅ **SÍ** |
| `#wildChipMetal` | 🥫 Metal | **Metal** | **97.5%** | ⚪ Caneca Blanca · Aprovechables | 488.9 ms | ✅ **SÍ** |
| `#wildChipMisc` | 🗑️ Basura Miscelánea | **Residuos Varios (Trash)** | **81.5%** | ⚫ Caneca Negra · No Aprovechables | 478.4 ms | ✅ **SÍ** |
| `#wildChipPaper` | 📄 Papel | **Papel (Paper)** | **99.6%** | ⚪ Caneca Blanca · Aprovechables | 497.5 ms | ✅ **SÍ** |
| `#wildChipPlastic` | 🧴 Plástico | **Plástico (Plastic)** | **93.0%** | ⚪ Caneca Blanca · Aprovechables | 488.5 ms | ✅ **SÍ** |
| `#wildChipTextile` | 👕 Residuos Textiles | **Textiles (Ropa / Telas)** | **99.8%** | ⚫ Caneca Negra · No Aprovechables | 493.3 ms | ✅ **SÍ** |
| `#wildChipVegetation` | 🌿 Vegetación y Jardinería | **Vegetación / Jardinería** | **99.8%** | 🟢 Caneca Verde · Orgánicos | 530.5 ms | ✅ **SÍ** |
| `#wildChipMedicamentos` | 💊 Medicamentos | **Medicamentos** *(renombrado)* | **86.9%** | 🔴 Caneca Roja · Peligrosos | 503.9 ms | ✅ **SÍ** |

- **Porcentaje de Acierto en Prueba Interactiva Chromium:** **100% (11/11)** con confianza media de **94.3%** (todas superando ampliamente el umbral mínimo del 75%).

### 3.2 Evaluación Masiva en Lote de 132 Fotos "In the Wild" Nuevas (`pruebas_wild_chromium/raw/`)

Además de la prueba interactiva en Chromium, se evaluó un banco ciego de **132 imágenes reales descargadas de internet** (12 por categoría):

| Categoría Problemática | Aciertos Exactos (19 Clases) | Acierto por Caneca Colombia | Confianza Promedio | Confusión Previa Mitigada |
| :--- | :---: | :---: | :---: | :--- |
| **Metal** | **12 / 12 (100.0%)** | 100.0% (Blanca) | **95.5%** | Eliminada confusión con pilas/bombillas/RAEE |
| **Vegetation** | **12 / 12 (100.0%)** | 100.0% (Verde) | **99.5%** | Confianza subió de ~30% inestable a **99.5%** |
| **Plastic** | **11 / 12 (91.7%)** | 100.0% (Blanca) | **89.1%** | Eliminada confusión con aceites químicos/bombillas |
| **Paper** | **11 / 12 (91.7%)** | 100.0% (Blanca) | **93.6%** | De 0% de detección a **91.7%** en todo escenario |
| **Textile_Trash** | **10 / 12 (83.3%)** | 91.7% (Negra) | **99.7%** | Eliminada desviación a residuos sanitarios |
| **Food_Organics** | **10 / 12 (83.3%)** | **100.0% (Verde)** | **93.2%** | 0% clasificado como medicamentos o tetra pack |
| **Medicamentos** | **10 / 12 (83.3%)** | 91.7% (Roja) | **94.7%** | Etiqueta unificada a `Medicamentos` |
| **Cardboard** | **9 / 12 (75.0%)** | 91.7% (Blanca) | **95.1%** | **0% de confusión con `Madera_Procesada`** |
| **Miscellaneous_Trash** | **8 / 12 (66.7%)** | 75.0% (Negra) | **86.7%** | Recuperación drástica de sensibilidad |
| **Glass** | **8 / 12 (66.7%)** | 75.0% (Blanca) | **88.7%** | Reconoce transparencias sin irse a aceites/bombillas |

---

## 4. Resumen de Cambios en Frontend, Etiquetas y Móvil

1. **Renombrado a `Medicamentos`:** Actualizado en `aplicaciones/server.py`, `momento tres/semana 13/flutter_app_residuos/lib/classifier.dart`, notebooks y generadores de presentación.
2. **Corrección del Botón de Cámara Web (`aplicaciones/index.html`):** Implementado `abrirCamaraWebRTC()` usando `navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } } })` con visor `<video id="cameraVideo">` en vivo, botón de captura instantánea en `<canvas>`, alternancia de cámara frontal/trasera y cierre limpio de stream.
3. **Homologación de Estilos UI en App Móvil (`momento tres/semana 13/flutter_app_residuos/lib/main.dart`):** Rediseñada la app Flutter con el mismo fondo radial azul oscuro (`#1E293B` → `#0F172A`), tarjetas glassmórficas, barra de latencia SLA, tarjetas de Caneca Colombia (Blanca, Verde, Negra, Roja) y cuadrícula 2×2 de la Resolución 2184 de 2019. APK recompilado en `aplicaciones/EcoClasificador_Colombia_Res2184.apk`.

