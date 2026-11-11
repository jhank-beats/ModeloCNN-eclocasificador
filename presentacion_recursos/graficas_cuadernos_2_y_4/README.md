# 🖼️ Catálogo de Gráficas Extraídas — Cuaderno 2 y Cuaderno 4
## Proyecto: EcoClasificador Móvil con Visión Artificial y Edge AI (19 Clases)
### Ubicación: `presentacion_recursos/graficas_cuadernos_2_y_4/`

Todas las figuras generadas durante los experimentos y ejecuciones de los **Cuadernos 2 y 4** han sido extraídas en alta resolución y organizadas con nombres descriptivos para su fácil inserción en diapositivas (PowerPoint, Canva, Google Slides) o pósteres científicos.

---

## 📌 Cuaderno 2: Entrenamiento del Modelo Padre (MobileNetV2 Baseline - 70.88%)

1. **`cuaderno2_distribucion_clases_dataset.png`**  
   * **Descripción:** Gráfico de barras con la distribución inicial de imágenes por categoría en las 19 clases.  
   * **Uso en diapositivas:** Muestra el desbalance de clases inicial y la composición del dataset.

2. **`cuaderno2_muestras_data_augmentation.png`**  
   * **Descripción:** Mosaico visual de ejemplos tras aplicar transformaciones de aumento de datos (*flips*, rotaciones leves y zoom).  
   * **Uso en diapositivas:** Explica la estrategia inicial para mitigar el sobreajuste.

3. **`cuaderno2_matriz_confusion_baseline_mobilenetv2.png`**  
   * **Descripción:** Matriz de confusión del modelo baseline MobileNetV2 sobre el conjunto de test.  
   * **Uso en diapositivas:** Evidencia el "techo de cristal" del 70.88% y las confusiones críticas en clases como Aceites, Pilas y Bombillas.

4. **`cuaderno2_prueba_inferencia_muestras.png`**  
   * **Descripción:** Inferencia visual con predicciones y barras de confianza sobre muestras reales del test set.  
   * **Uso en diapositivas:** Demuestra el comportamiento visual del modelo base.

---

## 🚀 Cuaderno 4: Optimización Avanzada hacia el 92.49% (EfficientNetV2 + Focal + Ensamble)

### Versiones Renderizadas en Celda:
5. **`cuaderno4_distribucion_pesos_clase_balanceados.png`**  
   * **Descripción:** Gráfico de pesos asignados a cada clase mediante `compute_class_weight("balanced")`.  
   * **Uso en diapositivas:** Justifica matemáticamente cómo se compensó el desbalance de las clases minoritarias.

6. **`cuaderno4_curvas_entrenamiento_loss_acc_lr.png`**  
   * **Descripción:** Evolución tripartita de Pérdida (Loss), Exactitud (Accuracy) y Tasa de Aprendizaje (LR Cosine Decay) a lo largo de las fases de Warmup y Fine-Tuning.  
   * **Uso en diapositivas:** Demuestra la convergencia suave y la ausencia de sobreajuste catastrófico.

7. **`cuaderno4_matriz_confusion_efficientnetv2_base.png`**  
   * **Descripción:** Matriz de confusión normalizada tras la Parte I (EfficientNetV2-B0 con AdamW, MixUp y CutMix - 82.66%).  
   * **Uso en diapositivas:** Muestra el primer gran salto de exactitud (+11.8% sobre el baseline).

8. **`cuaderno4_comparativa_f1_score_cuaderno2_vs_cuaderno4.png`**  
   * **Descripción:** Gráfico comparativo de barras clase por clase del F1-Score (Cuaderno 2 MobileNetV2 vs. Cuaderno 4 EfficientNetV2).  
   * **Uso en diapositivas:** Demuestra las ganancias masivas en clases difíciles como *Aceites (+33%)*, *Pilas (+23%)* y *Bombillas (+25%)*.

9. **`cuaderno4_galeria_muestras_ruido_web_detectadas.png`**  
   * **Descripción:** Mosaico de las 12 muestras con mayor nivel de ruido patológico detectadas por *Confident Learning* (estanterías, diagramas, microscopios).  
   * **Uso en diapositivas:** Demuestra el rigor metodológico al limpiar imágenes fuera de dominio del scraping web.

10. **`cuaderno4_matriz_confusion_final_ensamble_92acc.png`**  
    * **Descripción:** Matriz de confusión normalizada definitiva del Ensamble Calibrado (80% Focal + 20% Base con Multi-TTA y Temperature Scaling).  
    * **Uso en diapositivas:** **La diapositiva estrella.** Evidencia la exactitud final del **92.49%**, con la diagonal principal en verde intenso y Top-2 de **98.24%**.

---

### Versiones Exportadas en Ultra Alta Definición (300 DPI):
Para impresiones de pósteres o diapositivas a pantalla completa en 4K/1080p:
* **`cuaderno4_curvas_entrenamiento_hd_300dpi.png`** (188 KB)
* **`cuaderno4_matriz_confusion_base_hd_300dpi.png`** (852 KB)
* **`cuaderno4_comparativa_f1_hd_300dpi.png`** (143 KB)
* **`cuaderno4_muestras_ruido_hd_300dpi.png`** (1.1 MB)
* **`cuaderno4_matriz_confusion_final_ensamble_hd_300dpi.png`** (828 KB)

