# EcoClasificador Móvil - Clasificación de Residuos (19 Clases)
## Universidad Cooperativa de Colombia (UCC)
**Estudiante:** Juan Camilo Murillo Moncaleano  
**Docente:** Ing. Óscar Díaz Triana  
**Normativa:** Resolución 2184 de 2019 de Colombia (Canecas Blanca, Verde, Negra y Roja)

### Estructura de Entregables del Proyecto

* **[notebooks/](file:///home/jhank/Desktop/ModeloCNN/notebooks/)**:
  * [S15_Auditoria_MurilloMoncaleano.ipynb](file:///home/jhank/Desktop/ModeloCNN/notebooks/S15_Auditoria_MurilloMoncaleano.ipynb): Cuaderno ejecutable y con salidas de la Semana 15 (evaluación con 3 semillas, métricas por clase, matriz de confusión, curvas PR y ROC, umbral de confianza de 0.70, pruebas de robustez, prueba de campo en la región y Grad-CAM de 10 aciertos y 10 errores).

* **[docs/](file:///home/jhank/Desktop/ModeloCNN/docs/)**:
  * [S15_Informe_Auditoria_MurilloMoncaleano.pdf](file:///home/jhank/Desktop/ModeloCNN/docs/S15_Informe_Auditoria_MurilloMoncaleano.pdf): Informe de auditoría en formato Normas APA 7 (3 páginas en PDF).
  * [S15_PruebaCampo_MurilloMoncaleano.csv](file:///home/jhank/Desktop/ModeloCNN/docs/S15_PruebaCampo_MurilloMoncaleano.csv): Registro de las 57 fotografías de la prueba de campo evaluadas en el modelo móvil TFLite (`archivo, clase_real, prediccion, confianza, tiempo_respuesta_ms`).
  * [S15_Fotos_PruebaCampo_MurilloMoncaleano.zip](file:///home/jhank/Desktop/ModeloCNN/docs/S15_Fotos_PruebaCampo_MurilloMoncaleano.zip): Archivo comprimido con las 57 fotos de la prueba de campo en la región ([pruebas_campo_region/](file:///home/jhank/Desktop/ModeloCNN/pruebas_campo_region/)).

* **[ENTREGABLES_FINALES/](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/)**:
  * Copia consolidada de los archivos de entrega final (`S15_Auditoria_MurilloMoncaleano.ipynb`, `S15_Informe_Auditoria_MurilloMoncaleano.pdf`, `S15_PruebaCampo_MurilloMoncaleano.csv`, `S15_Fotos_PruebaCampo_MurilloMoncaleano.zip`, `Exposicion_Final_EcoClasificador_92acc.pptx`, `Exposicion_Final_EcoClasificador_92acc.pdf` y `posterElectivaIII.pdf`). También en esta carpeta quedó el apk descargable y offline

* **[aplicaciones/](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/)**:
  * [EcoClasificador_Colombia_Res2184.apk](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/EcoClasificador_Colombia_Res2184.apk): Aplicación móvil Android compilada con el modelo TFLite FP16.
  * [server.py](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/server.py) e [index.html](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/index.html): Servidor e interfaz web del proyecto.

* **[momento_1_y_2_base/](file:///home/jhank/Desktop/ModeloCNN/momento_1_y_2_base/) y [momento tres/](file:///home/jhank/Desktop/ModeloCNN/momento%20tres/)**:
  * `1_Pipeline_Dataset_10k_Imagenes.ipynb`: Construcción y balanceo del dataset de 19 clases.
  * `2_Entrenamiento_Modelo_19Clases.ipynb`: Entrenamiento base inicial.
  * `3_Defensa_Tecnica_Regularizacion.ipynb`: Regularización L1/L2 y optimizadores.
  * `4_Optimizacion_Avanzada_EfficientNetV2_90Acc.ipynb` (y `_EJECUTADO.ipynb`): Fine-Tuning con `EfficientNetV2-B0` y Focal Loss.
  * `5_Comparativa_Ensamble_vs_Modelo_Liviano_Destilado.ipynb`: Comparación entre ensambles (promedio, ponderado, híbrido) y modelo liviano TFLite.
  * `semana 11/` y `semana 12/`: Transfer Learning comparativo y cuantización a TensorFlow Lite.
  * `semana 13/flutter_app_residuos/`: Código fuente de la aplicación móvil en Flutter.
