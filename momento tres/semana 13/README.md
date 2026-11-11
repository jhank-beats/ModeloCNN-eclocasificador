# Semana 13 · Despliegue Móvil y Presentación en la Bienal UCC 2026
## Fase 2: Despliegue e Integración Mobile (28 al 30 de Octubre)

Este directorio contiene la aplicación móvil completa, la medición de rendimiento en hardware real, las pruebas de campo y el póster institucional.

---

### Estructura de Entregables

```
momento tres/semana 13/
├── flutter_app_residuos/                                       # Código fuente de la App en Flutter
│   ├── pubspec.yaml                                            # Configuración de dependencias (tflite_flutter, image)
│   └── lib/
│       ├── classifier.dart                                     # Servicio de preprocesamiento, inferencia y categorización
│       └── main.dart                                           # Interfaz de usuario (Cámara/Galería, Top-3, SLA, Alertas)
│
├── web_mobile_pwa/                                             # Simulador Móvil Interactivo (Offline / Stand Bienal)
│   └── index.html                                              # PWA interactiva para demostración en vivo
│
├── fotos_prueba_campo/                                         # Muestras fotográficas de campo (20 imágenes)
│   ├── campo_01_Aceites_Quimicos.jpg
│   ├── campo_02_Bombillas.jpg
│   └── ... (20 fotos de prueba regional)
│
├── informe_pruebas_campo_y_dispositivos.md                     # Informe técnico detallado de hardware y campo
├── Poster_Bienal_2026_Clasificador_Residuos.pdf                # Póster formal de presentación en la Bienal
├── generar_prueba_campo_20fotos.py                             # Script de recolección y auditoría de campo
└── generate_poster_bienal.py                                   # Script de generación del póster PDF
```

---

### Resultados Clave

1. **Inferencia 100% Offline:**
   - La aplicación móvil funciona sin conexión a internet, sin servidores externos y con un peso de solo **2.7 MB** (`modelo_mobilenetv2_dynamic_range.tflite`).

2. **Medición en Dispositivos Reales:**
   - **Xiaomi Redmi Note 11:** **14.6 ms** promedio (~68 FPS).
   - **Samsung Galaxy A13:** **24.1 ms** promedio (~41 FPS).
   - Ambos cumplen sobradamente el SLA ($< 2,000\text{ ms}$).

3. **Prueba de Campo (20 Fotos Nuevas):**
   - **Efectividad:** **90.0% de aciertos** (18 de 20 muestras).
   - **Transparencia Científica:** La app emite advertencia de baja confianza cuando el porcentaje es menor al 50%.
