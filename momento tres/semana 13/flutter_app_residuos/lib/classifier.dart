import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class PredictionResult {
  final String label;
  final String labelEs;
  final double confidence;
  final String caneca;
  final String canecaTitle;
  final String canecaIcon;
  final int canecaColorValue;
  final String category;
  final String recommendation;

  PredictionResult({
    required this.label,
    required this.labelEs,
    required this.confidence,
    required this.caneca,
    required this.canecaTitle,
    required this.canecaIcon,
    required this.canecaColorValue,
    required this.category,
    required this.recommendation,
  });
}

class ClassifierService {
  static const String modelPath = "assets/modelo_clasificador_19_clases.tflite";
  static const String labelsPath = "assets/labels.txt";
  static const int inputSize = 180;
  static const double confidenceThreshold = 0.40;

  Interpreter? _interpreter;
  List<String> _labels = [];

  bool get isLoaded => _interpreter != null && _labels.isNotEmpty;

  static const Map<String, String> _nombresEspanol = {
    'Aceites_Quimicos': 'Aceites y Químicos',
    'Bombillas': 'Bombillas (Mercurio / LED)',
    'Cardboard': 'Cartón (Cardboard)',
    'Escombros_Ceramica': 'Escombros y Cerámica',
    'Food_Organics': 'Residuos Orgánicos / Alimentos',
    'Glass': 'Vidrio (Glass)',
    'Madera_Procesada': 'Madera Procesada',
    'Medicamentos': 'Medicamentos Vencidos',
    'Metal': 'Metal',
    'Miscellaneous_Trash': 'Residuos Varios (Trash)',
    'Neumaticos_Gomas': 'Neumáticos y Gomas',
    'Paper': 'Papel (Paper)',
    'Pilas_Baterias': 'Pilas y Baterías',
    'Plastic': 'Plástico (Plastic)',
    'RAEE_Electronicos': 'RAEE (Electrónicos)',
    'Residuos_Sanitarios': 'Residuos Sanitarios',
    'Tetra_Pack': 'Tetra Pak / Envases Asépticos',
    'Textile_Trash': 'Textiles (Ropa / Telas)',
    'Vegetation': 'Vegetación / Jardinería',
  };

  Future<void> loadModel() async {
    try {
      // 1. Cargar etiquetas ordenadas
      final labelsData = await rootBundle.loadString(labelsPath);
      _labels = labelsData.split('\n').where((s) => s.trim().isNotEmpty).toList();

      // 2. Cargar modelo TFLite de alta precisión
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      _interpreter?.allocateTensors();
      print("✅ Modelo TFLite de alta precisión cargado con éxito. Clases: ${_labels.length}");
    } catch (e) {
      print("⚠️ Intentando fallback con modelo base TFLite: $e");
      try {
        final options = InterpreterOptions()..threads = 4;
        _interpreter = await Interpreter.fromAsset("assets/modelo_mobilenetv2_dynamic_range.tflite", options: options);
        _interpreter?.allocateTensors();
      } catch (errFallback) {
        print("❌ Error fatal cargando TFLite Interpreter: $errFallback");
      }
    }
  }

  Future<Map<String, dynamic>> classifyImage(File imageFile) async {
    if (_interpreter == null) {
      throw Exception("El modelo TFLite no ha sido inicializado.");
    }

    final stopwatch = Stopwatch()..start();

    // 1. Decodificar imagen
    final rawBytes = await imageFile.readAsBytes();
    final decodedImage = img.decodeImage(rawBytes);
    if (decodedImage == null) {
      throw Exception("No se pudo decodificar la imagen seleccionada.");
    }

    // 2. Redimensionar preservando aspecto con padding a 180x180 px
    final resizedImage = img.copyResize(
      decodedImage,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    // 3. Preprocesamiento: valores flotantes [0.0, 255.0]
    var input = Float32List(1 * inputSize * inputSize * 3);
    var bufferIndex = 0;
    for (var y = 0; y < inputSize; y++) {
      for (var x = 0; x < inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        input[bufferIndex++] = pixel.r.toDouble();
        input[bufferIndex++] = pixel.g.toDouble();
        input[bufferIndex++] = pixel.b.toDouble();
      }
    }

    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);

    // 4. Tensor de Salida [1, 19]
    final outputTensor = List.filled(1 * _labels.length, 0.0).reshape([1, _labels.length]);

    // 5. Inferencia TFLite en CPU del dispositivo
    _interpreter!.run(inputTensor, outputTensor);
    stopwatch.stop();
    final latencyMs = stopwatch.elapsedMilliseconds;

    // 6. Extraer probabilidades y ordenar Top-3
    final List<double> rawScores = List<double>.from(outputTensor[0]);
    List<MapEntry<int, double>> scoredIndices = [];
    for (int i = 0; i < rawScores.length; i++) {
      scoredIndices.add(MapEntry(i, rawScores[i]));
    }
    scoredIndices.sort((a, b) => b.value.compareTo(a.value));

    List<PredictionResult> top3 = [];
    for (int i = 0; i < 3 && i < scoredIndices.length; i++) {
      final index = scoredIndices[i].key;
      final conf = scoredIndices[i].value;
      final label = _labels[index];
      final labelEs = _nombresEspanol[label] ?? label.replaceAll("_", " ");
      final info = _getCanecaInfoColombia(label);

      top3.add(PredictionResult(
        label: label,
        labelEs: labelEs,
        confidence: conf,
        caneca: info["caneca"]!,
        canecaTitle: info["canecaTitle"]!,
        canecaIcon: info["canecaIcon"]!,
        canecaColorValue: int.parse(info["canecaColorValue"]!),
        category: info["category"]!,
        recommendation: info["recommendation"]!,
      ));
    }

    final topPrediction = top3.first;
    final isLowConfidence = topPrediction.confidence < confidenceThreshold;

    return {
      "top3": top3,
      "latencyMs": latencyMs,
      "isLowConfidence": isLowConfidence,
      "warningMessage": isLowConfidence
          ? "⚠️ Nivel de confianza moderado (${(topPrediction.confidence * 100).toStringAsFixed(1)}%). Verifique el residuo visualmente o acerque la toma."
          : null,
    };
  }

  // 🇨🇴 Normativa Colombia: Resolución 2184 de 2019 + Manejo RESPEL
  Map<String, String> _getCanecaInfoColombia(String label) {
    switch (label) {
      // ⚪ CANECA BLANCA: Residuos Aprovechables (Limpio y Seco)
      case 'Plastic':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. La botella o envase debe estar limpio, vacío y seco (sin líquidos ni comida). Aplástalo para reducir volumen."
        };
      case 'Cardboard':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. Cajas y cartón secos y desarmados. Si está impregnado de grasa o comida, debe ir a la caneca negra."
        };
      case 'Glass':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. Botellas y frascos de vidrio limpios y sin romper. Retira tapas metálicas o plásticas (deposítalas por separado)."
        };
      case 'Paper':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. Papel de archivo, periódicos y revistas limpios y secos. No arrugar excesivamente."
        };
      case 'Metal':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. Latas de bebidas, conservas y tapas metálicas enjuagadas y escurridas para evitar malos olores."
        };
      case 'Tetra_Pack':
        return {
          "caneca": "Blanca",
          "canecaTitle": "Caneca Blanca · Aprovechables",
          "canecaIcon": "⚪",
          "canecaColorValue": "0xFFFFFFFF",
          "category": "Aprovechable (Reciclable)",
          "recommendation": "Depositar en CANECA BLANCA. Enjuagar ligeramente, escurrir, desplegar las solapas laterales y aplastar para optimizar espacio."
        };

      // 🟢 CANECA VERDE: Residuos Orgánicos Aprovechables
      case 'Food_Organics':
        return {
          "caneca": "Verde",
          "canecaTitle": "Caneca Verde · Orgánicos Aprovechables",
          "canecaIcon": "🟢",
          "canecaColorValue": "0xFF10B981",
          "category": "Orgánico Aprovechable (Compostaje)",
          "recommendation": "Depositar en CANECA VERDE. Cáscaras de frutas, sobras de comida, restos de café y cáscaras de huevo. Apto para compostaje y biofertilizantes."
        };
      case 'Vegetation':
        return {
          "caneca": "Verde",
          "canecaTitle": "Caneca Verde · Orgánicos Aprovechables",
          "canecaIcon": "🟢",
          "canecaColorValue": "0xFF10B981",
          "category": "Orgánico Aprovechable (Restos Vegetales)",
          "recommendation": "Depositar en CANECA VERDE. Césped cortado, hojas secas, ramas y residuos de poda. No mezclar con bolsas plásticas ni piedras."
        };

      // ⚫ CANECA NEGRA: Residuos No Aprovechables / Ordinarios
      case 'Miscellaneous_Trash':
        return {
          "caneca": "Negra",
          "canecaTitle": "Caneca Negra · No Aprovechables",
          "canecaIcon": "⚫",
          "canecaColorValue": "0xFF475569",
          "category": "No Aprovechable (Ordinario)",
          "recommendation": "Depositar en CANECA NEGRA. Servilletas usadas, papel higiénico, cartón sucio con grasa y empaques no reciclables. Destino: Relleno sanitario."
        };
      case 'Textile_Trash':
        return {
          "caneca": "Negra",
          "canecaTitle": "Caneca Negra · Textil Degradado",
          "canecaIcon": "⚫",
          "canecaColorValue": "0xFF475569",
          "category": "No Aprovechable / Textil",
          "recommendation": "Depositar en CANECA NEGRA. Prendas de vestir muy deterioradas o manchadas. Si la prenda está limpia y usable, priorizar la donación."
        };
      case 'Madera_Procesada':
        return {
          "caneca": "Negra",
          "canecaTitle": "Caneca Negra · Madera Procesada",
          "canecaIcon": "⚫",
          "canecaColorValue": "0xFF475569",
          "category": "No Aprovechable / Especial",
          "recommendation": "Depositar en CANECA NEGRA (o Servicio de Voluminosos). Maderas tratadas con barniz, pinturas o clavos. Para piezas grandes, programar ruta de voluminosos."
        };

      // 🔴 CANECA ROJA: Residuos Peligrosos, Posconsumo y Manejo Especial (RESPEL)
      case 'Pilas_Baterias':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Posconsumo Pilas",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Residuo Peligroso (Metales Pesados)",
          "recommendation": "¡Peligro! Depositar en CANECA ROJA o Punto Posconsumo 'Pilas con el Ambiente' en centros comerciales. Nunca mezclar con basura ordinaria."
        };
      case 'Medicamentos':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Posconsumo Fármacos",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Residuo Peligroso / Farmacéutico",
          "recommendation": "Depositar en CANECA ROJA o Contenedor 'Punto Azul' en droguerías. Medicamentos vencidos o tabletas no consumidas. Nunca arrojarlos al inodoro."
        };
      case 'Aceites_Quimicos':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Químicos y Aceites (RESPEL)",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Residuo Peligroso / Químico",
          "recommendation": "Depositar en CANECA ROJA o Punto Limpio de Aceites. 1 litro de aceite contamina 1.000 litros de agua. Envasar cerrado y llevar a punto limpio."
        };
      case 'Bombillas':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Posconsumo Iluminación",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Residuo Peligroso (Mercurio)",
          "recommendation": "Depositar en CANECA ROJA o Buzón Posconsumo 'Lúmina'. Tubos y bombillos contienen vapor de mercurio. Entregar intactos, sin quebrar."
        };
      case 'RAEE_Electronicos':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Residuos RAEE",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Manejo Especial (Electrónico)",
          "recommendation": "Depositar en CANECA ROJA o Centros de Acopio RAEE (EcoCómputo / Red Verde). Cables, cargadores y aparatos electrónicos."
        };
      case 'Residuos_Sanitarios':
        return {
          "caneca": "Roja",
          "canecaTitle": "Caneca Roja · Riesgo Biológico",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Residuo Peligroso / Biológico",
          "recommendation": "Depositar en CANECA ROJA / Bolsa Roja Sellada. Material con fluidos corporales, gasas o jeringas. Requiere tratamiento e incineración controlada."
        };
      case 'Escombros_Ceramica':
        return {
          "caneca": "Roja",
          "canecaTitle": "Manejo Especial · Escombros RCD",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Manejo Especial (RCD)",
          "recommendation": "Manejo Especial / CANECA ROJA. Ladrillos, cerámicas y cemento. Solicitar servicio especial de recolección de escombros de la ciudad."
        };
      case 'Neumaticos_Gomas':
        return {
          "caneca": "Roja",
          "canecaTitle": "Manejo Especial · Posconsumo Llantas",
          "canecaIcon": "🔴",
          "canecaColorValue": "0xFFEF4444",
          "category": "Manejo Especial (Posconsumo)",
          "recommendation": "Manejo Especial / CANECA ROJA. Entregar en servitecas o puntos del programa 'Rueda Verde' para su reutilización en mezclas asfálticas."
        };

      default:
        return {
          "caneca": "Negra",
          "canecaTitle": "Caneca Negra · No Aprovechable",
          "canecaIcon": "⚫",
          "canecaColorValue": "0xFF475569",
          "category": "Residuo Ordinario",
          "recommendation": "Disponer con la recolección ordinaria de residuos no aprovechables."
        };
    }
  }

  void dispose() {
    _interpreter?.close();
  }
}
