#!/usr/bin/env python3
"""
=============================================================================
SCRIPT DE AUDITORÍA Y CONTEO DEL DATASET AMPLIADO (19 CATEGORÍAS)
=============================================================================
Universidad Cooperativa de Colombia (UCC) - Momento 3 / Semana 11
Proyecto de Impacto Regional: Clasificador Inteligente de Residuos

Objetivo:
- Demostrar el volumen por clase del dataset unificado (>= 300 imgs/clase).
- Identificar cuáles son las 10 clases propias recolectadas en la región
  y cuáles son las 9 clases base del benchmark RealWaste.
- Generar reporte tabular y gráfico para sustentar ante el docente.
=============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Directorio del dataset consolidado
DATASET_DIR = "dataset_completo"
if not os.path.exists(DATASET_DIR):
    # Fallback si se ejecuta desde subcarpetas
    DATASET_DIR = os.path.join("..", "..", "dataset_completo")

CLASES_PROPIAS = [
    "Aceites_Quimicos",
    "Bombillas",
    "Escombros_Ceramica",
    "Madera_Procesada",
    "Medicamentos",
    "Neumaticos_Gomas",
    "Pilas_Baterias",
    "RAEE_Electronicos",
    "Residuos_Sanitarios",
    "Tetra_Pack"
]

CLASES_BASE_REALWASTE = [
    "Cardboard",
    "Food_Organics",
    "Glass",
    "Metal",
    "Miscellaneous_Trash",
    "Paper",
    "Plastic",
    "Textile_Trash",
    "Vegetation"
]

def auditar_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: No se encontró la carpeta del dataset en '{DATASET_DIR}'")
        return None

    categorias = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")])
    
    registros = []
    total_imgs = 0
    total_propias = 0
    total_base = 0

    print("=" * 78)
    print("📊 AUDITORÍA DEL DATASET AMPLIADO: VOLUMEN Y TAXONOMÍA REGIONAL (MOMENTO 3)")
    print("=" * 78)
    print(f"{'#':<3} | {'Categoría':<22} | {'Tipo':<16} | {'Total Imgs':<10} | {'Cumple Mín (300)':<16}")
    print("-" * 78)

    for idx, cat in enumerate(categorias, 1):
        ruta_cat = os.path.join(DATASET_DIR, cat)
        archivos = [f for f in os.listdir(ruta_cat) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        conteo = len(archivos)
        total_imgs += conteo
        
        if cat in CLASES_PROPIAS:
            tipo = "Propia (Región)"
            total_propias += conteo
        else:
            tipo = "Base RealWaste"
            total_base += conteo
            
        cumple = "✅ SÍ (>300)" if conteo >= 300 else "⚠️ NO (<300)"
        print(f"{idx:<3} | {cat:<22} | {tipo:<16} | {conteo:<10} | {cumple:<16}")
        
        registros.append({
            "Indice": idx,
            "Categoria": cat,
            "Tipo": tipo,
            "Total_Imagenes": conteo,
            "Cumple_Minimo": conteo >= 300
        })

    print("-" * 78)
    print(f"📌 RESUMEN:")
    print(f"   • Total Categorías:          {len(categorias)} (10 Propias Regionales + 9 Base RealWaste)")
    print(f"   • Imágenes Clases Propias:   {total_propias:,} ({total_propias/total_imgs*100:.1f}%)")
    print(f"   • Imágenes Clases Base:      {total_base:,} ({total_base/total_imgs*100:.1f}%)")
    print(f"   • TOTAL GENERAL DATASET:     {total_imgs:,} imágenes estandarizadas (180x180 px RGB)")
    print("=" * 78)

    df = pd.DataFrame(registros)
    
    # Generar gráfico de sustentación
    plt.figure(figsize=(14, 7))
    colores = ["#2E7D32" if t == "Propia (Región)" else "#1565C0" for t in df["Tipo"]]
    bars = plt.barh(df["Categoria"], df["Total_Imagenes"], color=colores, edgecolor="black", alpha=0.85)
    plt.axvline(300, color="red", linestyle="--", linewidth=1.8, label="Umbral Mínimo Requerido (300 imgs)")
    plt.axvline(1000, color="orange", linestyle=":", linewidth=1.5, label="Meta Objetivo (1,000 imgs)")
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 12, bar.get_y() + bar.get_height()/2, f"{w:,}", va="center", fontsize=8.5, fontweight="bold")

    plt.xlabel("Número de Imágenes por Categoría", fontsize=11, fontweight="bold")
    plt.ylabel("Categorías de Residuos", fontsize=11, fontweight="bold")
    plt.title("Distribución del Dataset Ampliado: 10 Clases Propias Regionales vs. 9 Clases Base (UCC Momento 3)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    salida_grafica = "distribucion_dataset_19clases.png"
    plt.savefig(salida_grafica, dpi=180)
    plt.close()
    print(f"📈 Gráfico guardado exitosamente en: '{salida_grafica}'")
    return df

if __name__ == "__main__":
    auditar_dataset()
