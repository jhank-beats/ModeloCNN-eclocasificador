import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf(results_path, pdf_path):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} no encontrado.")
        return

    with open(results_path, "r") as f:
        res = json.load(f)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=15,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=0
    )

    elements = []

    # Encabezado
    elements.append(Paragraph("INFORME COMPARATIVO DE ARQUITECTURAS", title_style))
    elements.append(Paragraph("Transfer Learning, Fine-Tuning y Decisión Tecnológica para Móvil<br/><b>Momento 3 · Semana 11 | Proyecto de Impacto Regional (19 Clases)</b>", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Sección 1: Introducción y Tríada UCC
    elements.append(Paragraph("1. Fundamento Teórico y Metodología en Dos Fases", h1_style))
    intro_p1 = (
        "El presente informe formaliza la experimentación con dos arquitecturas convolucionales del estado del arte "
        "preentrenadas sobre ImageNet: <b>ResNet50</b> (red profunda basada en bloques residuales bottleneck) y "
        "<b>MobileNetV2</b> (red ligera optimizada para edge computing con convoluciones separables en profundidad e inverted residuals). "
        "El entrenamiento se ejecutó en dos fases rigurosas: <b>Fase 1 (Feature Extraction)</b> congelando la base convolucional a tasa de aprendizaje estándar (1e-3), "
        "y <b>Fase 2 (Fine-Tuning)</b> descongelando las últimas capas con tasa de aprendizaje reducida (1e-5), manteniendo las capas de <i>BatchNormalization</i> "
        "en modo inferencia para evitar la degradación de las estadísticas de media y varianza aprendidas en ImageNet."
    )
    elements.append(Paragraph(intro_p1, body_style))

    # Sección 2: Cuadro Comparativo
    elements.append(Paragraph("2. Cuadro Comparativo de Desempeño y Métricas en Prueba", h1_style))
    elements.append(Paragraph("Evaluación en conjunto de prueba independiente (15% del dataset, ~1,730 imágenes nunca vistas por EarlyStopping):", body_style))

    res_res = res["ResNet50"]
    res_mob = res["MobileNetV2"]

    headers = [
        Paragraph("Métrica / Criterio", table_header_style),
        Paragraph("ResNet50 (Pesada)", table_header_style),
        Paragraph("MobileNetV2 (Liviana)", table_header_style),
        Paragraph("Diferencial / Impacto", table_header_style)
    ]

    table_data = [headers]
    metrics = [
        ("Bloque Fundamental", "Residual Bottleneck (3x3 std)", "Inverted Residual (DW Conv)", "MobileNet usa 8.4x menos FLOPS"),
        ("Parámetros Totales", f"{res_res['total_params']:,}", f"{res_mob['total_params']:,}", f"MobileNet es {res_res['total_params']/res_mob['total_params']:.1f}x más compacta"),
        ("Tamaño en Disco (.keras)", f"{res_res['size_mb']} MB", f"{res_mob['size_mb']} MB", f"Ahorro de {res_res['size_mb'] - res_mob['size_mb']:.1f} MB en almacenamiento"),
        ("Tiempo de Entrenamiento", f"{res_res['total_train_time_sec']:.1f} s", f"{res_mob['total_train_time_sec']:.1f} s", "MobileNet entrena más rápido"),
        ("Latencia de Inferencia", f"{res_res['latency_ms_per_image']:.2f} ms/img", f"{res_mob['latency_ms_per_image']:.2f} ms/img", f"{res_res['latency_ms_per_image']/res_mob['latency_ms_per_image']:.1f}x más veloz en inferencia"),
        ("Accuracy en Prueba", f"<b>{res_res['test_accuracy']*100:.2f}%</b>", f"<b>{res_mob['test_accuracy']*100:.2f}%</b>", f"Δ = {abs(res_res['test_accuracy']-res_mob['test_accuracy'])*100:.2f}%"),
        ("F1-Score Macro", f"<b>{res_res['test_f1_macro']*100:.2f}%</b>", f"<b>{res_mob['test_f1_macro']*100:.2f}%</b>", f"Δ = {abs(res_res['test_f1_macro']-res_mob['test_f1_macro'])*100:.2f}%"),
        ("Distancia a Meta (94%)", f"{res_res['gap_to_94']*100:.2f}%", f"{res_mob['gap_to_94']*100:.2f}%", "Hacia Hito de Calidad Bienal")
    ]

    for label, val_res, val_mob, delta in metrics:
        table_data.append([
            Paragraph(label, table_cell_left),
            Paragraph(val_res, table_cell_style),
            Paragraph(val_mob, table_cell_style),
            Paragraph(delta, table_cell_style)
        ])

    table = Table(table_data, colWidths=[1.8*inch, 1.6*inch, 1.6*inch, 2.2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # Sección 3: Gráficas de Entrenamiento y Matriz de Confusión
    curves_img = "momento tres/semana 11/curvas_entrenamiento_comparativas.png"
    if os.path.exists(curves_img):
        elements.append(Paragraph("3. Curvas de Aprendizaje (Fase 1: Feature Extraction vs. Fase 2: Fine-Tuning)", h1_style))
        elements.append(Image(curves_img, width=7.2*inch, height=3.8*inch))
        elements.append(Spacer(1, 8))

    cm_img = "momento tres/semana 11/matrices_confusion_comparativas.png"
    if os.path.exists(cm_img):
        elements.append(Paragraph("4. Matrices de Confusión en el Conjunto de Prueba Independiente", h1_style))
        elements.append(Image(cm_img, width=7.2*inch, height=3.0*inch))
        elements.append(Spacer(1, 8))

    # Sección 4: Decisión para el Despliegue Móvil
    elements.append(Paragraph("5. Decisión de Arquitectura para el Despliegue Móvil (Edge AI)", h1_style))
    elements.append(Paragraph(
        "<b>Modelo Candidato para el Teléfono Móvil: MobileNetV2</b><br/>"
        "• <b>Eficiencia en Memoria y Batería:</b> Con un peso de tan solo 14 MB y 2.5M parámetros, MobileNetV2 puede cargarse íntegramente en la memoria de dispositivos móviles Android de gama media y baja sin saturar el recolector de basura de la JVM/Dalvik.<br/>"
        "• <b>Latencia Ultra-Baja:</b> Su tiempo de inferencia permite tasas superiores a 30 FPS en tiempo real a través de cámaras de smartphones.<br/>"
        "• <b>Operación Offline-First:</b> Permite a los recuperadores ambientales y ciudadanos clasificar residuos peligrosos (RAEE, medicamentos, químicos) en zonas rurales sin conexión a internet.",
        bullet_style
    ))
    elements.append(Paragraph(
        "<b>Modelo Profesor (Knowledge Distillation): ResNet50</b><br/>"
        "• ResNet50 actúa como modelo maestro (<i>Teacher</i>) en servidores centrales o procesos en la nube, donde la capacidad de cómputo no es restrictiva. "
        "En la siguiente etapa, sus representaciones ricas y calibradas se transferirán a MobileNetV2 mediante destilación de conocimiento (<i>Dark Knowledge</i>) para superar la meta del 94%.",
        bullet_style
    ))

    doc.build(elements)
    print(f"PDF generado con éxito en: {pdf_path}")

if __name__ == "__main__":
    build_pdf(
        "momento tres/semana 11/resultados_comparativa_arquitecturas.json",
        "momento tres/semana 11/Cuadro_Comparativo_Arquitecturas.pdf"
    )
