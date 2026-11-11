import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_poster(pdf_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=32,
        leftMargin=32,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    poster_title = ParagraphStyle(
        'PosterTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0D3B66"),
        alignment=1,
        spaceAfter=2
    )

    poster_subtitle = ParagraphStyle(
        'PosterSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#007A78"),
        alignment=1,
        spaceAfter=3
    )

    authors_style = ParagraphStyle(
        'Authors',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=8
    )

    box_title = ParagraphStyle(
        'BoxTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0D3B66"),
        spaceAfter=4
    )

    box_content = ParagraphStyle(
        'BoxContent',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3
    )

    bullet_content = ParagraphStyle(
        'BulletContent',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
        leftIndent=8,
        spaceAfter=2
    )

    table_th = ParagraphStyle(
        'TTH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_td = ParagraphStyle(
        'TTD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1A202C"),
        alignment=1
    )

    elements = []

    # Banner Header
    elements.append(Paragraph("SISTEMA MÓVIL OFFLINE PARA LA CLASIFICACIÓN DE RESIDUOS URBANOS Y PELIGROSOS CON DEEP LEARNING (TFLite)", poster_title))
    elements.append(Paragraph("BIENAL INSTITUCIONAL DE INVESTIGACIÓN E INNOVACIÓN UCC 2026 · MOMENTO 3", poster_subtitle))
    elements.append(Paragraph("<b>Electiva III: Visión Artificial y Machine Learning</b> | Universidad Cooperativa de Colombia", authors_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0D3B66"), spaceAfter=8))

    # Grid de 2 Columnas simulado con Tabla Principal
    # Columna Izquierda: 1. Problema Regional + 2. Dataset Ampliado + 3. Arquitectura y Destilación
    col1_content = []
    
    # 1. Problema Regional
    col1_content.append(Paragraph("1. PROBLEMA DE IMPACTO REGIONAL", box_title))
    col1_content.append(Paragraph(
        "En la región, la disposición inadecuada de <b>residuos peligrosos (RAEE, medicamentos, baterías, aceites químicos)</b> "
        "y hospitalarios en rellenos ordinarios genera contaminación de acuíferos y riesgos severos para los recuperadores de oficio. "
        "La falta de conectividad móvil en zonas de recolección impedía el uso de APIs en la nube. "
        "<b>Objetivo:</b> Crear una solución <i>Edge AI</i> que corra de manera autónoma, offline y en tiempo real en smartphones de gama media/baja.",
        box_content
    ))
    col1_content.append(Spacer(1, 4))

    # 2. Dataset
    col1_content.append(Paragraph("2. DATASET AMPLIADO (11,541 IMÁGENES)", box_title))
    col1_content.append(Paragraph(
        "• <b>19 Clases Normalizadas (180x180 px):</b> 9 de RealWaste + <b>10 Clases Propias Regionales</b>.<br/>"
        "• <b>Volumen Regional:</b> 6,789 fotos (58.8% del dataset) tomadas y curadas en campo (>300 fotos/clase).<br/>"
        "• <b>Partición Estratificada:</b> 70% Entrenamiento, 15% Validación, 15% Prueba Independiente (1,734 fotos).",
        bullet_content
    ))
    col1_content.append(Spacer(1, 4))

    # 3. Arquitectura y Cuantización
    col1_content.append(Paragraph("3. PIPELINE DE ENTRENAMIENTO Y TFLITE", box_title))
    col1_content.append(Paragraph(
        "• <b>Profesor (Teacher):</b> ResNet50 (24M params) entrenado en 2 fases con BatchNorm congelado.<br/>"
        "• <b>Estudiante Móvil:</b> MobileNetV2 afinado con <b>Keras Tuner</b> (10 ensayos de regularización).<br/>"
        "• <b>Conversión TFLite:</b> Rango Dinámico e INT8 con dataset representativo (reducción a solo <b>2.7 MB</b>).",
        bullet_content
    ))

    # Columna Derecha: 4. Resultados Técnicos + 5. La App Móvil + 6. Prueba de Campo y Conclusiones
    col2_content = []

    # 4. Resultados y Métricas
    col2_content.append(Paragraph("4. DESEMPEÑO Y BENCHMARK REAL", box_title))
    
    t_data = [
        [Paragraph("Métrica / Parámetro", table_th), Paragraph("Float32", table_th), Paragraph("TFLite Dynamic", table_th)],
        [Paragraph("Tamaño en Disco", table_td), Paragraph("9.2 MB", table_td), Paragraph("<b>2.7 MB (-71%)</b>", table_td)],
        [Paragraph("Test Accuracy", table_td), Paragraph("88.4%", table_td), Paragraph("<b>88.1%</b>", table_td)],
        [Paragraph("Latencia Móvil (CPU)", table_td), Paragraph("38.2 ms", table_td), Paragraph("<b>14.5 ms</b>", table_td)],
        [Paragraph("Throughput", table_td), Paragraph("26 FPS", table_td), Paragraph("<b>68 FPS</b>", table_td)],
        [Paragraph("Cumplimiento SLA", table_td), Paragraph("&lt; 2000 ms", table_td), Paragraph("<b>137x más rápido</b>", table_td)],
    ]
    t_table = Table(t_data, colWidths=[1.3*inch, 1.0*inch, 1.3*inch])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D3B66")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    col2_content.append(t_table)
    col2_content.append(Spacer(1, 4))

    # 5. La Aplicación Móvil
    col2_content.append(Paragraph("5. INTEGRACIÓN Y APLICATIVO MÓVIL", box_title))
    col2_content.append(Paragraph(
        "• <b>Offline-First:</b> Inferencia local sin consumo de datos ni dependencia de internet.<br/>"
        "• <b>Top-3 Honesto:</b> Muestra las 3 clases más probables con porcentajes y alerta de baja confianza (<50%).<br/>"
        "• <b>Guía de Disposición:</b> Clasificación por color de caneca (Blanco, Verde, Negro, Rojo posconsumo).",
        bullet_content
    ))
    col2_content.append(Spacer(1, 4))

    # 6. Prueba de Campo
    col2_content.append(Paragraph("6. PRUEBA DE CAMPO Y HONESTIDAD CIENTÍFICA", box_title))
    col2_content.append(Paragraph(
        "• <b>20 Fotografías Nuevas en la Región:</b> 17 aciertos (<b>85.0% de efectividad en campo</b>).<br/>"
        "• <b>Análisis de Confusión:</b> Errores detectados entre <i>Madera Procesada</i> y <i>Cartón Sucio</i> bajo sombras fuertes, "
        "y <i>Medicamentos en Blíster</i> confundidos con <i>Plástico</i>, demostrando la necesidad de capturas bien iluminadas.",
        box_content
    ))

    # Tabla contenedor de 2 columnas
    main_table = Table([[col1_content, col2_content]], colWidths=[3.7*inch, 3.7*inch])
    main_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('LINEBEFORE', (1, 0), (1, -1), 1, colors.HexColor("#E2E8F0")),
    ]))
    elements.append(main_table)
    elements.append(Spacer(1, 6))

    # Footer Institucional
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0D3B66"), spaceAfter=4))
    elements.append(Paragraph(
        "<b>Universidad Cooperativa de Colombia · Sede Regional</b> · Proyecto de Ciencia Aplicada e Innovación Tecnológica (Bienal 2026)",
        authors_style
    ))

    doc.build(elements)
    print(f"Póster generado con éxito en: {pdf_path}")

if __name__ == "__main__":
    create_poster("momento tres/semana 13/Poster_Bienal_2026_Clasificador_Residuos.pdf")
