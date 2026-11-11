# Momento 3 · Electiva III: Visión Artificial y Proyecto de Impacto Regional
## Rumbo a la Bienal (28 al 30 de Octubre)

Este directorio consolida las entregas académicas, cuadernos reproducibles, modelos y artefactos técnicos para el despliegue del sistema clasificador de residuos urbanos y peligrosos (19 clases, 11,541 imágenes).

---

### Estructura de Entregables

```
momento tres/
│
├── README.md                                                     # Índice general del Momento 3
│
├── semana 11/                                                    # Hito 1: Arquitecturas Avanzadas y Transfer Learning
│   ├── Transfer_Learning_FineTuning_ResNet50_vs_MobileNetV2.ipynb# Cuaderno reproducible (ResNet50 vs MobileNetV2)
│   ├── conteo_dataset.py                                         # Script de auditoría de volumen y clases propias
│   ├── distribucion_dataset_19clases.png                         # Gráfica de distribución por clase (11,541 imágenes)
│   ├── Cuadro_Comparativo_Arquitecturas.pdf                      # Informe formal comparativo (ResNet50 vs MobileNetV2)
│   ├── generate_notebook.py                                      # Generador automatizado del cuaderno
│   ├── generate_pdf_report.py                                    # Generador automatizado del informe PDF
│   └── run_experiments.py                                        # Script de entrenamiento en dos fases
│
└── semana 12/                                                    # Hito 2 (Final): Ajuste de Hiperparámetros y TFLite Móvil
    ├── Ajuste_Hiperparametros_Conversion_TFLite_19Clases.ipynb  # Cuaderno ejecutable de optimización y TFLite
    ├── labels.txt                                                # Archivo de etiquetas sincronizado (19 clases)
    ├── CONTRATO_APP_MOVIL.md                                     # Contrato formal con desarrolladores móviles
    ├── Informe_Tecnico_Optimizacion_TFLite_Bienal.pdf            # Informe técnico de cuantización y latencia
    ├── generate_semana12_notebook.py                             # Generador del cuaderno semana 12
    ├── generate_pdf_semana12.py                                  # Generador del informe PDF semana 12
    ├── pipeline_tflite_hparams.py                                # Pipeline de Keras Tuner y exportación a TFLite
    ├── modelo_mobilenetv2_float32.tflite                         # Modelo TFLite sin cuantizar (Float32)
    ├── modelo_mobilenetv2_dynamic_range.tflite                   # Modelo TFLite optimizado (Rango Dinámico ~2.7 MB)
    └── modelo_mobilenetv2_int8.tflite                            # Modelo TFLite calibrado en enteros (INT8)
```

---

### Resumen de Cumplimiento de Criterios

| Criterio Exigido | Implementación | Evidencia |
| :--- | :--- | :--- |
| **Volumen de Imágenes** | $\ge 300 - 500$ imágenes por clase | 11,541 imágenes totales (todas las 19 clases cumplen) |
| **Clases Propias** | 10 categorías de impacto regional | `conteo_dataset.py` y `distribucion_dataset_19clases.png` |
| **Comparación de Redes** | Red Pesada (ResNet50) vs Liviana (MobileNetV2) | Cuaderno Semana 11 y reporte comparativo |
| **Entrenamiento 2 Fases** | Fase 1 ($lr=10^{-3}$) y Fase 2 ($lr=10^{-5}$, BatchNorm congelado) | Código y curvas en Semana 11 |
| **Ajuste de Hiperparámetros**| Keras Tuner / Optuna ($\ge 10$ ensayos) | Pipeline Keras Tuner en Semana 12 |
| **Conversión TFLite** | 3 variantes: Float32, Rango Dinámico e INT8 | Archivos `.tflite` en Semana 12 |
| **SLA de Latencia Móvil** | Inferencia $< 2,000$ ms por imagen | Latencia real lograda: **~12 - 25 ms** |
| **Contrato Móvil** | Entrada, normalización, `labels.txt` y salida | `CONTRATO_APP_MOVIL.md` y `labels.txt` |
