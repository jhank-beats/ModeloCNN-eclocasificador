import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ResiduosApp());
}

class ResiduosApp extends StatelessWidget {
  const ResiduosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EcoClasificador Colombia',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B5E20),
          brightness: Brightness.light,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ClassifierService _classifier = ClassifierService();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  Map<String, dynamic>? _classificationResult;
  bool _isLoading = false;
  bool _isModelReady = false;

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    await _classifier.loadModel();
    setState(() {
      _isModelReady = _classifier.isLoaded;
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1080,
        maxHeight: 1080,
      );

      if (pickedFile == null) return;

      setState(() {
        _selectedImage = File(pickedFile.path);
        _isLoading = true;
        _classificationResult = null;
      });

      final result = await _classifier.classifyImage(_selectedImage!);

      setState(() {
        _classificationResult = result;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error durante la inferencia: $e")),
      );
    }
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "EcoClasificador Móvil",
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF1B5E20),
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showAboutDialog(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status bar
            Container(
              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
              decoration: BoxDecoration(
                color: _isModelReady ? Colors.green.shade50 : Colors.orange.shade50,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: _isModelReady ? Colors.green.shade300 : Colors.orange.shade300,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    _isModelReady ? Icons.check_circle : Icons.hourglass_top,
                    color: _isModelReady ? Colors.green : Colors.orange,
                    size: 18,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _isModelReady
                          ? "Edge AI Listo · 19 Clases · Res. 2184 Colombia"
                          : "Cargando motor TFLite en CPU...",
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: _isModelReady ? Colors.green.shade800 : Colors.orange.shade800,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Visor de Imagen
            Center(
              child: Container(
                width: double.infinity,
                height: 250,
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: _selectedImage != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(18),
                        child: Image.file(
                          _selectedImage!,
                          fit: BoxFit.contain,
                          width: double.infinity,
                        ),
                      )
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.recycling, size: 70, color: Colors.grey.shade400),
                          const SizedBox(height: 8),
                          Text(
                            "Toma o selecciona la foto de un residuo",
                            style: TextStyle(color: Colors.grey.shade600, fontSize: 14),
                          ),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 16),

            // Botones Cámara / Galería
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isModelReady && !_isLoading
                        ? () => _pickImage(ImageSource.camera)
                        : null,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text("Cámara"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1B5E20),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      elevation: 3,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isModelReady && !_isLoading
                        ? () => _pickImage(ImageSource.gallery)
                        : null,
                    icon: const Icon(Icons.photo_library),
                    label: const Text("Galería"),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: const Color(0xFF1B5E20),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      side: const BorderSide(color: Color(0xFF1B5E20), width: 1.5),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),

            if (_isLoading) ...[
              const Center(child: CircularProgressIndicator(color: Color(0xFF1B5E20))),
              const SizedBox(height: 8),
              const Center(child: Text("Ejecutando inferencia neuronal local en el dispositivo...")),
            ],

            // Resultados de Clasificación
            if (_classificationResult != null) ...[
              _buildResultCard(_classificationResult!),
              const SizedBox(height: 20),
            ],

            // Panel Informativo Normativa Colombia (Siempre visible para guía)
            _buildNormativaGuia(),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(Map<String, dynamic> result) {
    final List<PredictionResult> top3 = result["top3"];
    final int latencyMs = result["latencyMs"];
    final bool isLowConfidence = result["isLowConfidence"];
    final String? warningMessage = result["warningMessage"];
    final PredictionResult best = top3.first;

    final Color canecaBorderColor = Color(best.canecaColorValue);
    final String canecaNombre = best.caneca.toLowerCase();

    Color canecaBgColor;
    if (canecaNombre == 'blanca') {
      canecaBgColor = Colors.grey.shade100;
    } else if (canecaNombre == 'verde') {
      canecaBgColor = Colors.green.shade50;
    } else if (canecaNombre == 'roja') {
      canecaBgColor = Colors.red.shade50;
    } else {
      canecaBgColor = Colors.blueGrey.shade50;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // SLA y Latencia
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              "Latencia Edge AI: ${latencyMs} ms",
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey.shade700),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: latencyMs < 2000 ? Colors.green.shade100 : Colors.orange.shade100,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                latencyMs < 2000 ? "SLA Cumplido (< 2s)" : "SLA Alerta",
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  color: latencyMs < 2000 ? Colors.green.shade800 : Colors.orange.shade900,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),

        if (isLowConfidence && warningMessage != null) ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.amber.shade100,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.amber.shade700),
            ),
            child: Row(
              children: [
                const Icon(Icons.warning_amber_rounded, color: Colors.amber, size: 24),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    warningMessage,
                    style: TextStyle(color: Colors.brown.shade900, fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],

        // Tarjeta Principal (Mejor Predicción)
        Card(
          elevation: 3,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Residuo Detectado:",
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade600, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        best.labelEs,
                        style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                      ),
                    ),
                    Text(
                      "${(best.confidence * 100).toStringAsFixed(1)}%",
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: isLowConfidence ? Colors.orange : const Color(0xFF1B5E20),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                // 🇨🇴 BANNER DESTACADO DE CANECA SEGÚN NORMATIVA COLOMBIA
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: canecaBgColor,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: canecaNombre == 'blanca' ? Colors.grey.shade400 : canecaBorderColor,
                      width: 2,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(best.canecaIcon, style: const TextStyle(fontSize: 22)),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: Colors.black.withAlpha(20),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: const Text(
                                    "🇨🇴 Res. 2184/2019 MinAmbiente",
                                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold),
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  best.canecaTitle,
                                  style: TextStyle(
                                    fontSize: 15,
                                    fontWeight: FontWeight.bold,
                                    color: canecaNombre == 'blanca' ? Colors.grey.shade900 : canecaBorderColor,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Text(
                        best.recommendation,
                        style: TextStyle(fontSize: 13, color: Colors.grey.shade900, height: 1.35),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),

        // Top 3 Clases Más Probables
        const Text(
          "Distribución Top-3 Probabilidades:",
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        ...top3.map((pred) {
          final Color barColor = Color(pred.canecaColorValue);
          final bool isWhite = pred.caneca.toLowerCase() == 'blanca';

          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 4.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      "${pred.canecaIcon} ${pred.labelEs} (Caneca ${pred.caneca})",
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                    ),
                    Text(
                      "${(pred.confidence * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: pred.confidence,
                    minHeight: 8,
                    backgroundColor: Colors.grey.shade200,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      isWhite ? Colors.blueGrey.shade400 : barColor,
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  Widget _buildNormativaGuia() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Text("🇨🇴", style: TextStyle(fontSize: 18)),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  "Código de Colores Oficial (Res. 2184 de 2019)",
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _itemCanecaGuia("⚪ Caneca Blanca", "Aprovechables: Plástico, vidrio, metales, papel, cartón y tetra pak limpios y secos.", Colors.grey.shade800),
          _itemCanecaGuia("🟢 Caneca Verde", "Orgánicos aprovechables: Restos de comida, cáscaras, desechos agrícolas y jardín.", const Color(0xFF10B981)),
          _itemCanecaGuia("⚫ Caneca Negra", "No aprovechables: Papel higiénico, servilletas usadas, cartón grasiento y residuos comunes.", const Color(0xFF475569)),
          _itemCanecaGuia("🔴 Caneca Roja", "Peligrosos y Posconsumo: Pilas, medicamentos, aceites, bombillas, RAEE y biológicos.", const Color(0xFFEF4444)),
        ],
      ),
    );
  }

  Widget _itemCanecaGuia(String titulo, String desc, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 5, right: 8),
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: const TextStyle(fontSize: 12, color: Colors.black87),
                children: [
                  TextSpan(text: "$titulo: ", style: TextStyle(fontWeight: FontWeight.bold, color: color)),
                  TextSpan(text: desc),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("EcoClasificador Móvil"),
        content: const Text(
          "Proyecto de Clasificación Automatizada de Residuos Urbanos y Peligrosos.\n\n"
          "• Modelo: EfficientNetV2-B0 Cuantizado FP16\n"
          "• Exactitud: >92% (19 Categorías)\n"
          "• Normativa: Resolución 2184 de 2019 (MinAmbiente Colombia)\n"
          "• Inferencia: 100% Offline en CPU móvil (< 200 ms)\n"
          "• Privacidad: Las fotos no salen del teléfono",
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("Cerrar"),
          ),
        ],
      ),
    );
  }
}
