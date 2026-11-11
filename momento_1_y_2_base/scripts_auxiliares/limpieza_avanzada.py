#!/usr/bin/env python3
"""
=============================================================================
SCRIPT DE LIMPIEZA AUTOMÁTICA AVANZADA (ANTI-CATÁLOGO, TEXTO Y DIBUJOS)
=============================================================================
Analiza las imágenes ya descargadas en 'dataset_suplementario' y detecta:
1. Fotos de catálogo / tienda aisladas sobre fondo blanco puro (>70% borde blanco).
2. Afiches, diagramas y folletos con texto estructurado residual.
3. Dibujos digitales, esquemas o vectores que escaparon al primer filtro.

Uso:
  ./.venv/bin/python limpieza_avanzada.py --modo scan     (Solo analiza y muestra reporte)
  ./.venv/bin/python limpieza_avanzada.py --modo limpiar  (Mueve las sospechosas a descartes)
=============================================================================
"""

import os
import sys
import shutil
import argparse
import cv2
import numpy as np

DATASET_DIR = "dataset_suplementario"
DESCARTES_DIR = "dataset_suplementario_descartes"

def es_producto_catalogo_fondo_blanco(img_bgr):
    """Detecta si el objeto está aislado sobre fondo blanco de estudio/catálogo."""
    h, w = img_bgr.shape[:2]
    # Extraer los 4 bordes exteriores
    top = img_bgr[0:3, :, :]
    bottom = img_bgr[-3:, :, :]
    left = img_bgr[:, 0:3, :]
    right = img_bgr[:, -3:, :]

    borde_pixeles = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])
    
    # Calcular promedio de brillo de cada píxel del borde
    brillo_borde = np.mean(borde_pixeles, axis=1)
    blanco_ratio = np.count_nonzero(brillo_borde > 242) / float(len(brillo_borde))
    
    if blanco_ratio > 0.75:
        return True, f"fondo_blanco_catalogo_{blanco_ratio:.2f}"
    return False, "OK"

def tiene_texto_o_diagrama(img_bgr):
    """Detecta bloques de texto tipográfico o encabezados publicitarios."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Gradientes horizontales típicos de líneas de texto
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = (mag > 60).astype(np.uint8)
    
    kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_text)
    
    # Relación de cajas de texto
    ratio = np.count_nonzero(morph) / float(morph.size)
    if ratio > 0.38:
        return True, f"texto_denso_{ratio:.2f}"
        
    return False, "OK"

def es_dibujo_o_vector_sintetico(img_bgr):
    """Detecta ilustraciones, animaciones y cliparts por saturación artificial y bordes duros."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    
    # Si la saturación es excesivamente uniforme y artificial
    sat_mean = np.mean(sat)
    sat_std = np.std(sat)
    
    # Gráfico plano con colores saturados
    if sat_mean > 120 and sat_std < 22:
        return True, "color_plano_vector"
        
    return False, "OK"

def evaluar_sospechosa(img_bgr):
    es_cat, r_cat = es_producto_catalogo_fondo_blanco(img_bgr)
    if es_cat:
        return True, r_cat
        
    es_txt, r_txt = tiene_texto_o_diagrama(img_bgr)
    if es_txt:
        return True, r_txt
        
    es_dib, r_dib = es_dibujo_o_vector_sintetico(img_bgr)
    if es_dib:
        return True, r_dib
        
    return False, "OK"

def main():
    parser = argparse.ArgumentParser(description="Limpiador avanzado de dataset")
    parser.add_argument("--modo", choices=["scan", "limpiar"], default="scan",
                        help="'scan' para ver cuántas detecta; 'limpiar' para moverlas a descartes.")
    parser.add_argument("--categoria", type=str, default=None,
                        help="Categoría específica a procesar (ej. Aceites_Quimicos). Si no se indica, procesa todas.")
    args = parser.parse_args()

    print("=" * 75)
    print(f"🧹 ANÁLISIS DE LIMPIEZA AVANZADA (MODO: {args.modo.upper()})")
    if args.categoria:
        print(f"🎯 CATEGORÍA SELECCIONADA: {args.categoria}")
    print("=" * 75)

    if not os.path.exists(DATASET_DIR):
        print(f"Error: No existe '{DATASET_DIR}'.")
        return

    if args.categoria:
        if not os.path.exists(os.path.join(DATASET_DIR, args.categoria)):
            print(f"Error: La categoría '{args.categoria}' no existe en '{DATASET_DIR}'.")
            return
        categorias = [args.categoria]
    else:
        categorias = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')])
    
    total_revisadas = 0
    total_sospechosas = 0
    reporte = {}

    for cat in categorias:
        ruta_cat = os.path.join(DATASET_DIR, cat)
        archivos = sorted([f for f in os.listdir(ruta_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        desc_cat = os.path.join(DESCARTES_DIR, cat)
        if args.modo == "limpiar":
            os.makedirs(desc_cat, exist_ok=True)
            
        sospechosas_cat = []
        for f in archivos:
            ruta_f = os.path.join(ruta_cat, f)
            im = cv2.imread(ruta_f)
            if im is None:
                continue
            
            es_mala, motivo = evaluar_sospechosa(im)
            if es_mala:
                sospechosas_cat.append((ruta_f, motivo, f))
                if args.modo == "limpiar":
                    dest_f = os.path.join(desc_cat, f"avanzado_{motivo}_{f}")
                    shutil.move(ruta_f, dest_f)
                    
        total_revisadas += len(archivos)
        total_sospechosas += len(sospechosas_cat)
        reporte[cat] = (len(archivos), len(sospechosas_cat))
        
        accion_txt = "Descartadas" if args.modo == "limpiar" else "Detectadas"
        pct = (len(sospechosas_cat) / len(archivos) * 100) if archivos else 0
        print(f"📁 {cat:<22} | Total: {len(archivos):4d} | {accion_txt}: {len(sospechosas_cat):3d} ({pct:4.1f}%)")

    print("-" * 75)
    pct_total = (total_sospechosas / total_revisadas * 100) if total_revisadas else 0
    print(f"📊 TOTAL GENERAL REVISADO: {total_revisadas} imágenes")
    print(f"🚨 TOTAL SOSPECHOSAS (CATÁLOGO/TEXTO/DIBUJO): {total_sospechosas} ({pct_total:.1f}%)")
    print("=" * 75)
    
    if args.modo == "scan":
        print("\n💡 Consejo: Para mover automáticamente estas imágenes sospechosas a la carpeta de descartes, ejecuta:")
        print("   ./.venv/bin/python limpieza_avanzada.py --modo limpiar\n")

if __name__ == "__main__":
    main()
