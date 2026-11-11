# Proyecto de Visión Artificial · Clasificador de Residuos (19 Clases)
## Universidad Cooperativa de Colombia (UCC) · Bienal de Investigación 2026

Este repositorio contiene la canalización completa del ciclo de vida CRISP-ML(Q) para la clasificación de residuos urbanos y de manejo especial mediante redes neuronales convolucionales profundas y despliegue *Edge AI* en dispositivos móviles.

---

###  Estructura del Proyecto

* **[ENTREGABLES_FINALES/](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/)**:
  * `EcoClasificador_Bienal_UCC.apk`: Aplicación móvil Android para inferencia 100% offline.
  * `posterElectivaIII.pdf`: Póster institucional presentado en la Bienal.
  * `metodologia_diagram_en.png`: Diagrama de flujo metodológico.
  * `CONTRATO_APP_MOVIL.md`: Especificación técnica de integración (entradas, normalización y salidas).
  * `informe_pruebas_campo_y_dispositivos.md`: Resultados de campo en 20 fotos y medición de latencia en teléfonos reales.

* **[aplicaciones/](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/)**:
  * `app_19clases.py`: Aplicativo de escritorio/servidor para inferencia con el modelo de 19 clases.
  * `app_9clases_base.py`: Aplicativo base inicial de 9 clases (RealWaste).

* **[modelos/](file:///home/jhank/Desktop/ModeloCNN/modelos/)**:
  * `modelo_mobilenetv2_dynamic_range.tflite`: Modelo optimizado en formato TFLite (2.7 MB).
  * `labels.txt`: Mapeo oficial de las 19 clases sincronizado con la salida de la red.
  * `modelo_clasificador_19_clases.keras`: Modelo maestro entrenado en Keras.
  * `modelo_profesor_vgg16.keras`: Modelo profesor de destilación de conocimiento.

* **[momento tres/](file:///home/jhank/Desktop/ModeloCNN/momento%20tres/)**:
  * `semana 11/`: Transfer Learning y Fine-Tuning comparativo (ResNet50 vs MobileNetV2).
  * `semana 12/`: Optimización de hiperparámetros y cuantización TFLite.
  * `semana 13/`: Código fuente de la app Flutter (`flutter_app_residuos/`), PWA web y fotos de campo.

* **[momento_1_y_2_base/](file:///home/jhank/Desktop/ModeloCNN/momento_1_y_2_base/)**:
  * Cuadernos de preparación del dataset (`1_Pipeline_Dataset_10k_Imagenes.ipynb`).
  * Entrenamiento inicial (`2_Entrenamiento_Modelo_19Clases.ipynb`).
  * Defensa técnica de optimizadores y regularización L1/L2 (`3_Defensa_Tecnica_Regularizacion.ipynb`).
  * Scripts auxiliares de depuración y descarga.

* **[dataset_completo/](file:///home/jhank/Desktop/ModeloCNN/dataset_completo/)**:
  * 11,541 imágenes catalogadas en 19 clases (180x180 px RGB).

* **[poster/](file:///home/jhank/Desktop/ModeloCNN/poster/)**:
  * Archivos gráficos, capturas y póster final de la Bienal.
