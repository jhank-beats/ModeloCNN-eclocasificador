import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'classifier.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0F172A),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const ResiduosApp());
}

class ResiduosApp extends StatelessWidget {
  const ResiduosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EcoClasificador · Normativa Colombia Res. 2184',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00C853),
          secondary: Color(0xFF38BDF8),
          surface: Color(0xFF1E293B),
          error: Color(0xFFEF4444),
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

  // Paleta homologada con la interfaz Web (aplicaciones/index.html)
  static const Color kBgTop = Color(0xFF1E293B);
  static const Color kBgBottom = Color(0xFF0F172A);
  static const Color kCardBg = Color(0xD91E293B); // rgba(30, 41, 59, 0.85)
  static const Color kInnerDarkBg = Color(0x990F172A); // rgba(15, 23, 42, 0.6)
  static const Color kAccentGreen = Color(0xFF00C853);
  static const Color kPrimaryGreenDark = Color(0xFF1B5E20);
  static const Color kPrimaryGreenLight = Color(0xFF2E7D32);
  static const Color kTextMain = Color(0xFFF8FAFC);
  static const Color kTextMuted = Color(0xFF94A3B8);
  static const Color kBorderSubtle = Color(0x1AFFFFFF); // rgba(255,255,255,0.1)

  @override
  void initState() {
    super.initState();
    _initModel();
  }

  Future<void> _initModel() async {
    await _classifier.loadModel();
    if (!mounted) return;
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

      if (!mounted) return;
      setState(() {
        _classificationResult = result;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: const Color(0xFFEF4444),
          content: Text(
            "Error durante la inferencia: $e",
            style: const TextStyle(color: Colors.white),
          ),
        ),
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
    final int latencyMs = _classificationResult != null
        ? (_classificationResult!["latencyMs"] as int)
        : 0;

    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment.topCenter,
            radius: 1.35,
            colors: [kBgTop, kBgBottom],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 490),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // 1. Header idéntico a la versión Web
                    _buildWebHeader(context),
                    const SizedBox(height: 16),

                    // 2. Main Preview Card
                    _buildMainPreviewCard(latencyMs),
                    const SizedBox(height: 16),

                    // 3. Warning Box si confianza baja
                    if (_classificationResult != null &&
                        (_classificationResult!["isLowConfidence"] as bool) &&
                        _classificationResult!["warningMessage"] != null) ...[
                      _buildWarningBox(_classificationResult!["warningMessage"] as String),
                      const SizedBox(height: 16),
                    ],

                    // 4. Tarjeta de Resultados y Caneca Colombia
                    if (_classificationResult != null) ...[
                      _buildResultCard(_classificationResult!),
                      const SizedBox(height: 16),
                    ],

                    // 5. Guía Resumida del Código de Colores Colombia (Cuadrícula 2x2)
                    _buildNormativaPanel(),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildWebHeader(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      decoration: BoxDecoration(
        color: kCardBg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kBorderSubtle),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(76),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Flexible(
            child: Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 8,
              runSpacing: 4,
              children: [
                const Text(
                  "♻️ EcoClasificador",
                  style: TextStyle(
                    fontSize: 19,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                  decoration: BoxDecoration(
                    color: const Color(0x2600C853),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0x4D00C853)),
                  ),
                  child: const Text(
                    "92.5% Exactitud",
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: kAccentGreen,
                    ),
                  ),
                ),
              ],
            ),
          ),
          GestureDetector(
            onTap: () => _showAboutDialog(context),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _isModelReady ? kAccentGreen : const Color(0xFFF59E0B),
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: _isModelReady ? kAccentGreen : const Color(0xFFF59E0B),
                            blurRadius: 8,
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      _isModelReady ? "Normativa Col. 2184" : "Cargando IA...",
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: _isModelReady ? kAccentGreen : const Color(0xFFF59E0B),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0x2638BDF8),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0x4D38BDF8)),
                  ),
                  child: const Text(
                    "ℹ️ Edge AI Offline",
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF38BDF8),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMainPreviewCard(int latencyMs) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kCardBg,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: kBorderSubtle),
      ),
      child: Column(
        children: [
          // Visor de Imagen (.image-box)
          Container(
            width: double.infinity,
            height: 250,
            decoration: BoxDecoration(
              color: _selectedImage != null
                  ? const Color(0xFF0B1120)
                  : kInnerDarkBg,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(
                color: const Color(0x26FFFFFF),
                width: 1.5,
              ),
            ),
            child: _selectedImage != null
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Image.file(
                      _selectedImage!,
                      fit: BoxFit.contain,
                      width: double.infinity,
                      height: double.infinity,
                    ),
                  )
                : Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text("📸", style: TextStyle(fontSize: 44)),
                        const SizedBox(height: 8),
                        const Text(
                          "Toma una foto en vivo con tu cámara o sube una imagen de tu galería.",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: kTextMuted,
                            fontSize: 13.5,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 5,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0x1A38BDF8),
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(color: const Color(0x4038BDF8)),
                          ),
                          child: const Text(
                            "⚡ Procesamiento neuronal 100% local en tu teléfono",
                            style: TextStyle(
                              fontSize: 11.5,
                              color: Color(0xFF38BDF8),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
          const SizedBox(height: 14),

          // Barra de Métricas (.metrics-bar)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: const Color(0x800F172A),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                RichText(
                  text: TextSpan(
                    style: const TextStyle(fontSize: 13, color: kTextMain),
                    children: [
                      const TextSpan(text: "Inferencia: "),
                      TextSpan(
                        text: "$latencyMs ms",
                        style: const TextStyle(fontWeight: FontWeight.w800),
                      ),
                    ],
                  ),
                ),
                if (_isLoading)
                  const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      color: kAccentGreen,
                    ),
                  )
                else
                  RichText(
                    text: TextSpan(
                      style: const TextStyle(fontSize: 13, color: kTextMain),
                      children: [
                        const TextSpan(text: "SLA: "),
                        TextSpan(
                          text: latencyMs <= 2000 ? "Óptimo" : "Alerta",
                          style: TextStyle(
                            fontWeight: FontWeight.w800,
                            color: latencyMs <= 2000
                                ? kAccentGreen
                                : const Color(0xFFF59E0B),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Botones de Acción (.btn-group)
          Row(
            children: [
              Expanded(
                child: GestureDetector(
                  onTap: _isModelReady && !_isLoading
                      ? () => _pickImage(ImageSource.camera)
                      : null,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [kPrimaryGreenDark, kPrimaryGreenLight],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: const [
                        BoxShadow(
                          color: Color(0x661B5E20),
                          blurRadius: 12,
                          offset: Offset(0, 4),
                        ),
                      ],
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text("📷", style: TextStyle(fontSize: 15)),
                        SizedBox(width: 6),
                        Text(
                          "Cámara",
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: GestureDetector(
                  onTap: _isModelReady && !_isLoading
                      ? () => _pickImage(ImageSource.gallery)
                      : null,
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 13),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF1E3A8A), Color(0xFF2563EB)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: const [
                        BoxShadow(
                          color: Color(0x592563EB),
                          blurRadius: 12,
                          offset: Offset(0, 4),
                        ),
                      ],
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text("🖼️", style: TextStyle(fontSize: 15)),
                        SizedBox(width: 6),
                        Text(
                          "Subir Foto",
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildWarningBox(String warningMessage) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0x26F59E0B),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x4DF59E0B)),
      ),
      child: Row(
        children: [
          const Text("⚠️", style: TextStyle(fontSize: 18)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              warningMessage,
              style: const TextStyle(
                color: Color(0xFFFCD34D),
                fontSize: 13,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultCard(Map<String, dynamic> result) {
    final List<PredictionResult> top3 = result["top3"];
    final PredictionResult best = top3.first;
    final String canecaNombre = best.caneca.toLowerCase();

    Color borderColor;
    Color titleColor;
    Color iconBgColor;
    Color iconTextColor;
    Color gradientTint;

    if (canecaNombre == 'blanca') {
      borderColor = const Color(0x80FFFFFF);
      titleColor = const Color(0xFFFFFFFF);
      iconBgColor = const Color(0xFFFFFFFF);
      iconTextColor = const Color(0xFF0F172A);
      gradientTint = const Color(0x26FFFFFF);
    } else if (canecaNombre == 'verde') {
      borderColor = const Color(0xFF10B981);
      titleColor = const Color(0xFF34D399);
      iconBgColor = const Color(0xFF10B981);
      iconTextColor = const Color(0xFFFFFFFF);
      gradientTint = const Color(0x3810B981);
    } else if (canecaNombre == 'roja') {
      borderColor = const Color(0xFFEF4444);
      titleColor = const Color(0xFFF87171);
      iconBgColor = const Color(0xFFEF4444);
      iconTextColor = const Color(0xFFFFFFFF);
      gradientTint = const Color(0x38EF4444);
    } else {
      borderColor = const Color(0xFF64748B);
      titleColor = const Color(0xFFE2E8F0);
      iconBgColor = const Color(0xFF1E293B);
      iconTextColor = const Color(0xFFF8FAFC);
      gradientTint = const Color(0x4D475569);
    }

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: kCardBg,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: kBorderSubtle),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Top Prediction (.top-prediction)
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Expanded(
                child: Text(
                  best.labelEs,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
              Text(
                "${(best.confidence * 100).toStringAsFixed(1)}%",
                style: const TextStyle(
                  fontSize: 21,
                  fontWeight: FontWeight.w800,
                  color: kAccentGreen,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),

          // 🇨🇴 Tarjeta Caneca Colombia (.caneca-card)
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [gradientTint, const Color(0xF21E293B)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: borderColor, width: 2),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withAlpha(89),
                  blurRadius: 20,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Barra superior brillante de 4px
                  Container(
                    height: 4,
                    color: canecaNombre == 'blanca'
                        ? Colors.white
                        : borderColor,
                  ),
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 46,
                              height: 46,
                              decoration: BoxDecoration(
                                color: iconBgColor,
                                borderRadius: BorderRadius.circular(14),
                                border: canecaNombre == 'negra'
                                    ? Border.all(color: const Color(0x40FFFFFF))
                                    : null,
                              ),
                              alignment: Alignment.center,
                              child: Text(
                                best.canecaIcon,
                                style: TextStyle(
                                  fontSize: 22,
                                  color: iconTextColor,
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 8,
                                      vertical: 2,
                                    ),
                                    decoration: BoxDecoration(
                                      color: const Color(0x59000000),
                                      borderRadius: BorderRadius.circular(6),
                                    ),
                                    child: const Text(
                                      "🇨🇴 NORMATIVA COLOMBIA · RES. 2184/2019",
                                      style: TextStyle(
                                        fontSize: 10,
                                        fontWeight: FontWeight.w700,
                                        color: kTextMuted,
                                        letterSpacing: 0.5,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    best.canecaTitle,
                                    style: TextStyle(
                                      fontSize: 17,
                                      fontWeight: FontWeight.w800,
                                      color: titleColor,
                                      height: 1.2,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 12,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0xA60F172A),
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(color: const Color(0x14FFFFFF)),
                          ),
                          child: RichText(
                            text: TextSpan(
                              style: const TextStyle(
                                fontSize: 13.5,
                                color: Color(0xFFE2E8F0),
                                height: 1.45,
                              ),
                              children: [
                                const TextSpan(
                                  text: "Sugerencia de Disposición: ",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w700,
                                    color: Colors.white,
                                  ),
                                ),
                                TextSpan(text: best.recommendation),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Distribución Top-3 Probabilidades Calibradas
          const Text(
            "DISTRIBUCIÓN TOP-3 PROBABILIDADES CALIBRADAS:",
            style: TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: kTextMuted,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 10),
          ...top3.map((pred) {
            final String itemCaneca = pred.caneca.toLowerCase();
            Color dotColor = const Color(0xFF64748B);
            if (itemCaneca == 'blanca') dotColor = const Color(0xFFFFFFFF);
            if (itemCaneca == 'verde') dotColor = const Color(0xFF10B981);
            if (itemCaneca == 'roja') dotColor = const Color(0xFFEF4444);

            return Padding(
              padding: const EdgeInsets.only(bottom: 10.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Flexible(
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 9,
                              height: 9,
                              margin: const EdgeInsets.only(right: 6),
                              decoration: BoxDecoration(
                                color: dotColor,
                                shape: BoxShape.circle,
                                boxShadow: [
                                  BoxShadow(color: dotColor, blurRadius: 6),
                                ],
                              ),
                            ),
                            Flexible(
                              child: Text(
                                "${pred.labelEs} (Caneca ${pred.caneca})",
                                style: const TextStyle(
                                  fontSize: 13,
                                  color: kTextMain,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Text(
                        "${(pred.confidence * 100).toStringAsFixed(1)}%",
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w800,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 5),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: pred.confidence.clamp(0.0, 1.0),
                      minHeight: 6,
                      backgroundColor: const Color(0x1AFFFFFF),
                      valueColor: AlwaysStoppedAnimation<Color>(dotColor),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildNormativaPanel() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kCardBg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: kBorderSubtle),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "🇨🇴 CÓDIGO DE COLORES NACIONAL (RESOLUCIÓN 2184 DE 2019)",
            style: TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: kTextMuted,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _buildCanecaMiniCard(
                  titulo: "⚪ Caneca Blanca",
                  colorTitulo: Colors.white,
                  borderColor: Colors.white,
                  desc: "Aprovechables: Plástico, cartón, vidrio, papel, metales y tetra pak limpios y secos.",
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _buildCanecaMiniCard(
                  titulo: "🟢 Caneca Verde",
                  colorTitulo: const Color(0xFF34D399),
                  borderColor: const Color(0xFF10B981),
                  desc: "Orgánicos aprovechables: Restos de comida, cáscaras, desechos agrícolas y poda de jardín.",
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _buildCanecaMiniCard(
                  titulo: "⚫ Caneca Negra",
                  colorTitulo: const Color(0xFFCBD5E1),
                  borderColor: const Color(0xFF94A3B8),
                  desc: "No aprovechables: Papel higiénico, servilletas usadas, cartón grasiento y residuos comunes.",
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _buildCanecaMiniCard(
                  titulo: "🔴 Caneca Roja / RESPEL",
                  colorTitulo: const Color(0xFFF87171),
                  borderColor: const Color(0xFFEF4444),
                  desc: "Peligrosos y Posconsumo: Pilas, medicamentos, aceites, bombillas, RAEE y biológicos.",
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCanecaMiniCard({
    required String titulo,
    required Color colorTitulo,
    required Color borderColor,
    required String desc,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0x8C0F172A),
        borderRadius: BorderRadius.circular(14),
        border: Border(
          left: BorderSide(color: borderColor, width: 3),
          top: const BorderSide(color: Color(0x14FFFFFF)),
          right: const BorderSide(color: Color(0x14FFFFFF)),
          bottom: const BorderSide(color: Color(0x14FFFFFF)),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            titulo,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: colorTitulo,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            desc,
            style: const TextStyle(
              fontSize: 11.5,
              color: kTextMuted,
              height: 1.35,
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
        backgroundColor: const Color(0xFF1E293B),
        title: const Text(
          "♻️ EcoClasificador Móvil",
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        content: const Text(
          "Clasificación Inteligente de Residuos Urbanos y Peligrosos.\n\n"
          "• Arquitectura: EfficientNetV2-B0 Optimizada\n"
          "• Cobertura: 19 Categorías Nacionales\n"
          "• Normativa: Resolución 2184 de 2019 (MinAmbiente Colombia)\n"
          "• Inferencia: 100% Offline en Dispositivo (Edge AI)\n"
          "• Privacidad: Las capturas se procesan localmente",
          style: TextStyle(color: Color(0xFFE2E8F0), fontSize: 13.5, height: 1.4),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text(
              "Cerrar",
              style: TextStyle(color: kAccentGreen, fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }
}
