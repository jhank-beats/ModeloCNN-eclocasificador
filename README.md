# ♻️ EcoClasificador Móvil · Visión Artificial y Edge AI (19 Clases)
## Universidad Cooperativa de Colombia (UCC) · Campus Ibagué-Espinal
**Autor:** Juan Camilo Murillo Moncaleano  
**Docente:** Prof. Óscar Díaz Triana  
**Normativa:** Resolución 2184 de 2019 de Colombia (Canecas Blanca, Verde, Negra y Roja/RESPEL)

---

### 📂 Estructura de Entregables del Repositorio

* **[notebooks/](file:///home/jhank/Desktop/ModeloCNN/notebooks/)**:
  * [S15_Auditoria_MurilloMoncaleano.ipynb](file:///home/jhank/Desktop/ModeloCNN/notebooks/S15_Auditoria_MurilloMoncaleano.ipynb): Cuaderno ejecutable y con salidas de la **Semana 15** (Error por clase, Matriz de Confusión, Curvas ROC/PR, Umbral «No estoy seguro» $\tau^*=0.72$, Validación con 3 semillas, Pruebas de Robustez física, Prueba de Campo en el Tolima y Explicabilidad Grad-CAM de 10 aciertos y 10 errores).

* **[docs/](file:///home/jhank/Desktop/ModeloCNN/docs/)**:
  * [S15_Informe_Auditoria_MurilloMoncaleano.pdf](file:///home/jhank/Desktop/ModeloCNN/docs/S15_Informe_Auditoria_MurilloMoncaleano.pdf): Informe técnico de auditoría avanzada (3 páginas en PDF).
  * [S15_PruebaCampo_MurilloMoncaleano.csv](file:///home/jhank/Desktop/ModeloCNN/docs/S15_PruebaCampo_MurilloMoncaleano.csv): Registro oficial de las 57 fotografías nuevas de campo probadas en el motor móvil TFLite (`archivo, clase_real, prediccion, confianza, tiempo_respuesta_ms`).
  * [S15_Fotos_PruebaCampo_MurilloMoncaleano.zip](file:///home/jhank/Desktop/ModeloCNN/docs/S15_Fotos_PruebaCampo_MurilloMoncaleano.zip): Evidencia fotográfica comprimida de las 57 muestras regionales ([pruebas_campo_tolima/](file:///home/jhank/Desktop/ModeloCNN/pruebas_campo_tolima/)).

* **[ENTREGABLES_FINALES/](file:///home/jhank/Desktop/ModeloCNN/ENTREGABLES_FINALES/)**:
  * Copia consolidada de todos los entregables de sustentación final (`S15_Auditoria_MurilloMoncaleano.ipynb`, `S15_Informe_Auditoria_MurilloMoncaleano.pdf`, `S15_PruebaCampo_MurilloMoncaleano.csv`, `S15_Fotos_PruebaCampo_MurilloMoncaleano.zip`, `Exposicion_Final_EcoClasificador_92acc.pptx`, `Exposicion_Final_EcoClasificador_92acc.pdf` y `posterElectivaIII.pdf`).

* **[aplicaciones/](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/)**:
  * [EcoClasificador_Colombia_Res2184.apk](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/EcoClasificador_Colombia_Res2184.apk): Aplicación móvil Android compilada con el modelo TFLite FP16 para inferencia 100% offline.
  * [server.py](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/server.py) e [index.html](file:///home/jhank/Desktop/ModeloCNN/aplicaciones/index.html): Servidor e interfaz web progresiva con captura de cámara en vivo por WebRTC e inferencia Multi-Crop TTA.

* **[momento_1_y_2_base/](file:///home/jhank/Desktop/ModeloCNN/momento_1_y_2_base/) y [momento tres/](file:///home/jhank/Desktop/ModeloCNN/momento%20tres/)**:
  * `1_Pipeline_Dataset_10k_Imagenes.ipynb`: Construcción y balanceo del dataset de 19 clases.
  * `2_Entrenamiento_Modelo_19Clases.ipynb`: Entrenamiento base inicial.
  * `3_Defensa_Tecnica_Regularizacion.ipynb`: Regularización L1/L2 y optimizadores.
  * `4_Optimizacion_Avanzada_EfficientNetV2_90Acc.ipynb` (y `_EJECUTADO.ipynb`): Fine-Tuning con `EfficientNetV2-B0` y *Focal Loss*.
  * `5_Comparativa_Ensamble_vs_Modelo_Liviano_Destilado.ipynb`: Comparación entre ensambles (promedio, ponderado, híbrido en cascada) y modelo liviano TFLite.
  * `semana 11/` y `semana 12/`: Transfer Learning comparativo y cuantización a TensorFlow Lite.
  * `semana 13/flutter_app_residuos/`: Código fuente de la aplicación móvil en Flutter.
