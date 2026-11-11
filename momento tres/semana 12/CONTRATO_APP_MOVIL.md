# Contrato Técnico de Integración con la Aplicación Móvil
## Clasificador de Residuos Urbanos y Peligrosos (19 Clases) - Hito Bienal

Este documento establece el contrato formal y exhaustivo entre el modelo de Machine Learning (`.tflite`) y los desarrolladores de la aplicación móvil (Flutter / Android Nativo Kotlin / iOS Swift / React Native).

---

### 1. Ficha Técnica del Artefacto TFLite Recomendado
* **Nombre de Archivo:** `modelo_mobilenetv2_dynamic_range.tflite`
* **Arquitectura Base:** MobileNetV2 preentrenado en ImageNet + Fine-Tuning en 19 clases
* **Método de Optimización:** Cuantización de Rango Dinámico (Post-Training Dynamic Range Quantization)
* **Tamaño en Disco:** **~2.7 MB** (reducción del 75% frente a Float32)
* **Consumo de RAM:** ~18 MB en memoria de ejecución
* **Latencia Promedio en CPU Móvil:** **~12 - 25 ms por inferencia**
* **Throughput:** ~40 - 80 FPS
* **Cumplimiento de SLA:** Cumple sobradamente la meta institucional ($< 2,000\text{ ms}$).

---

### 2. Especificación de Entrada (Input Tensor)

| Propiedad | Valor Requerido | Descripción |
| :--- | :--- | :--- |
| **Tensor Index** | `0` | Índice del buffer de entrada en el intérprete |
| **Shape** | `[1, 180, 180, 3]` | `[BatchSize, Height, Width, Channels]` |
| **Data Type** | `FLOAT32` | Número en coma flotante de 32 bits |
| **Formato de Color** | **RGB** | Canales en orden Rojo, Verde, Azul (0 a 255 inicial) |

#### ⚠️ Regla de Oro: Fórmula de Preprocesamiento y Normalización
La mayoría de fallos en producción móvil ocurren por aplicar un reescalado incorrecto. **NO normalizar dividiendo por 255.0 ($[0, 1]$)**. La red MobileNetV2 fue entrenada en el espacio normalizado $[-1.0, 1.0]$:

$$\text{pixel}_{\text{normalizado}} = \frac{\text{pixel}}{127.5} - 1.0$$

**Código de Preprocesamiento en Kotlin / Java:**
```kotlin
val inputBuffer = ByteBuffer.allocateDirect(1 * 180 * 180 * 3 * 4)
inputBuffer.order(ByteOrder.nativeOrder())

for (y in 0 until 180) {
    for (x in 0 until 180) {
        val pixel = bitmapResized.getPixel(x, y)
        val r = (Color.red(pixel) / 127.5f) - 1.0f
        val g = (Color.green(pixel) / 127.5f) - 1.0f
        val b = (Color.blue(pixel) / 127.5f) - 1.0f
        inputBuffer.putFloat(r)
        inputBuffer.putFloat(g)
        inputBuffer.putFloat(b)
    }
}
```

---

### 3. Especificación de Salida (Output Tensor)

| Propiedad | Valor Esperado | Descripción |
| :--- | :--- | :--- |
| **Tensor Index** | `0` | Índice del buffer de salida |
| **Shape** | `[1, 19]` | Array 2D con las probabilidades para cada clase |
| **Data Type** | `FLOAT32` | Probabilidades continuas tras función Softmax |
| **Suma del Vector** | $\sum_{i=0}^{18} P_i = 1.0$ | Permite umbralizar confianza directamente |

#### Lógica de Decisión y Predicción:
```kotlin
val outputBuffer = Array(1) { FloatArray(19) }
tfliteInterpreter.run(inputBuffer, outputBuffer)

val probabilities = outputBuffer[0]
var maxIdx = 0
var maxProb = probabilities[0]
for (i in 1 until probabilities.size) {
    if (probabilities[i] > maxProb) {
        maxProb = probabilities[i]
        maxIdx = i
    }
}

val predictedClass = labelsList[maxIdx]
val confidencePercentage = maxProb * 100.0f
```

---

### 4. Mapeo Canónico de Clases (`labels.txt`)
El orden en el archivo `labels.txt` es estricto y corresponde al índice del array de salida:

```text
0: Aceites_Quimicos
1: Bombillas
2: Cardboard
3: Escombros_Ceramica
4: Food_Organics
5: Glass
6: Madera_Procesada
7: Medicamentos
8: Metal
9: Miscellaneous_Trash
10: Neumaticos_Gomas
11: Paper
12: Pilas_Baterias
13: Plastic
14: RAEE_Electronicos
15: Residuos_Sanitarios
16: Tetra_Pack
17: Textile_Trash
18: Vegetation
```

---

### 5. Categorización Visual y Directrices de Reciclaje para la App
| Categoría | Clases Incluidas | Color UI Recomendado | Mensaje / Disposición Recomendada |
| :--- | :--- | :--- | :--- |
| **Residuos Peligrosos y RAEE** | `Aceites_Quimicos`, `Bombillas`, `Medicamentos`, `Pilas_Baterias`, `RAEE_Electronicos` | 🔴 Rojo (#E53E3E) | **Peligro:** Entregar exclusivamente en puntos limpios posconsumo autorizados. |
| **Residuos Sanitarios** | `Residuos_Sanitarios` | ⚫ Negro (#2D3748) | **No Aprovechable:** Disponer en bolsa negra sellada para relleno sanitario. |
| **Reciclables Tradicionales** | `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Tetra_Pack` | ⚪ Blanco (#EDF2F7) / Azul | **Aprovechable:** Limpio y seco en la caneca blanca o entrega a recuperador. |
| **Orgánicos y Vegetación** | `Food_Organics`, `Vegetation` | 🟢 Verde (#38A169) | **Compostaje:** Depositar en caneca verde para compostaje o biomasa. |
| **Especiales / Voluminosos** | `Escombros_Ceramica`, `Madera_Procesada`, `Neumaticos_Gomas`, `Textile_Trash` | 🟡 Ámbar (#D69E2E) | **Especial:** Solicitar servicio municipal de recolección de voluminosos. |
