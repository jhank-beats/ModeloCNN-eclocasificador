# EcoClasificador Móvil (19 Clases) — Clasificación Offline de Residuos

**Curso:** Electiva III (`A_Electiva_III_2620_6089`) — Momento 3 (Entrega Final)  
**Institución:** Universidad Cooperativa de Colombia (UCC)  
**Integrante:** Juan Camilo Murillo Moncaleano  
**Docente:** Ing. Óscar Díaz Triana  
**Normativa:** Resolución 2184 de 2019 de Colombia (Canecas Blanca, Verde, Negra y Roja/Especial)  
**Video Pitch (Demo en Vivo):** `[PEGAR_ENLACE_AL_VIDEO_PITCH]`

---

## 1. Problema y Valor Regional
En Colombia, la separación en la fuente bajo la **Resolución 2184 de 2019** presenta altos errores por desconocimiento ciudadano frente a empaques compuestos (Tetra Pak), textiles y residuos peligrosos/posconsumo (pilas, medicamentos vencidos, bombillas, aceites químicos, RAEE y sanitarios). Además, muchos puntos ecológicos escolares, universitarios y rurales carecen de conectividad estable para depender de APIs en la nube.

**EcoClasificador Móvil** resuelve este problema mediante una aplicación Android **100% offline** (funciona en modo avión) basada en `EfficientNetV2-B0` cuantizada a TensorFlow Lite (`FP16`), que clasifica **19 categorías de residuos** en tiempo real e indica el color exacto de caneca correspondiente.

---

## 2. Métricas Principales del Modelo

| Métrica / Artefacto | Valor Obtenido | Detalle |
|---|---|---|
| **Accuracy (3 semillas: 42, 101, 2026)** | **96.77% ± 0.53%** | Evaluación estratificada sobre conjunto de prueba |
| **F1-Score Macro (3 semillas)** | **96.81% ± 0.51%** | Desempeño balanceado en las 19 categorías |
| **Accuracy con Umbral ≥ 0.70** | **99.76%** | Acepta el 88.00% de muestras; 12.00% abstención segura (*"No estoy seguro"*) |
| **Prueba de Campo en la Región (57 fotos)** | **94.74%** (54/57) | Evaluado con intérprete TFLite FP16 en fotos amateurs reales |
| **Tamaño del Modelo TFLite (FP16)** | **11.35 MB** | `model/modelo_efficientnetv2b0_fp16.tflite` (5,948,771 parámetros) |
| **Latencia de Inferencia (CPU XNNPACK)** | **14.46 ms / imagen** | Promedio en prueba de campo (~69 FPS teóricos) |

---

## 3. Estructura del Repositorio

* **[notebooks/](file:///home/jhank/Desktop/ModeloCNN/notebooks/)**:
  * [Notebook_Final_MurilloMoncaleano.ipynb](file:///home/jhank/Desktop/ModeloCNN/notebooks/Notebook_Final_MurilloMoncaleano.ipynb): Cuaderno final consolidado y ejecutado (entrenamiento, destilación, conversión a TFLite y evaluación integral).
  * [S15_Auditoria_MurilloMoncaleano.ipynb](file:///home/jhank/Desktop/ModeloCNN/notebooks/S15_Auditoria_MurilloMoncaleano.ipynb): Auditoría técnica completa de Semana 15 (3 semillas, PR/ROC, umbral 0.70, robustez, campo y Grad-CAM).
* **[model/](file:///home/jhank/Desktop/ModeloCNN/model/)**:
  * `modelo_efficientnetv2b0_fp16.tflite` (11.35 MB — modelo principal desplegado en la app).
  * `modelo_mobilenetv2_dynamic_range.tflite` (2.8 MB — alternativa cuantizada dinámica).
  * `labels.txt` (19 categorías ordenadas).
* **[app/](file:///home/jhank/Desktop/ModeloCNN/app/)**:
  * Código fuente completo de la aplicación móvil en Flutter (`lib/`, `assets/`, `android/`, `pubspec.yaml`).
* **[docs/](file:///home/jhank/Desktop/ModeloCNN/docs/)**:
  * [Pitch_Deck_MurilloMoncaleano.pdf](file:///home/jhank/Desktop/ModeloCNN/docs/Pitch_Deck_MurilloMoncaleano.pdf): Presentación final (10 diapositivas).
  * [S15_Informe_Auditoria_MurilloMoncaleano.pdf](file:///home/jhank/Desktop/ModeloCNN/docs/S15_Informe_Auditoria_MurilloMoncaleano.pdf): Informe de auditoría en formato APA 7 (3 páginas).
  * `figuras_s15/`, `S15_PruebaCampo_MurilloMoncaleano.csv` y `S15_Fotos_PruebaCampo_MurilloMoncaleano.zip`.
* **[ENTREGABLES_FINALES/](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/)**:
  * Carpeta lista para envío con `App_EcoClasificador.apk`, `Pitch_Deck_MurilloMoncaleano.pdf`, `Notebook_Final_MurilloMoncaleano.ipynb` y documentos de auditoría.

---

## 4. Cómo Instalar y Ejecutar

### A. Aplicación Móvil Android (APK Offline)
1. Descargar [ENTREGABLES_FINALES/App_EcoClasificador.apk](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/App_EcoClasificador.apk).
2. Instalar en un teléfono Android (habilitar *Instalar aplicaciones de orígenes desconocidos* si el sistema lo solicita).
3. Abrir **EcoClasificador** (funciona completamente en **Modo Avión**), tomar una fotografía nueva del residuo y observar la categoría predicha, confianza, latencia y caneca recomendada.

### B. Compilar la App desde Código Fuente (Flutter)
```bash
cd app
flutter pub get
flutter build apk --release
```

### C. Ejecutar los Cuadernos en Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
jupyter notebook notebooks/Notebook_Final_MurilloMoncaleano.ipynb
```
