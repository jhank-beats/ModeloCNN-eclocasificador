#!/usr/bin/env python3
"""
Monitor en tiempo real del progreso del dataset de 10 clases suplementarias.
Muestra tabla con recuentos limpios, descartes y porcentaje hacia la meta de 1,000.
"""

import os
import sys

CARPETA_DATASET = "dataset_suplementario"
CARPETA_DESCARTES = "dataset_suplementario_descartes"
META = 1000

CLASES = [
    "Aceites_Quimicos", "Bombillas", "Escombros_Ceramica", "Madera_Procesada",
    "Medicamentos", "Neumaticos_Gomas", "Pilas_Baterias", "RAEE_Electronicos",
    "Residuos_Sanitarios", "Tetra_Pack"
]

def contar_archivos(carpeta):
    if not os.path.exists(carpeta):
        return 0
    return len([f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

def barra_progreso(actual, total, longitud=20):
    fraccion = min(1.0, actual / float(total))
    relleno = int(longitud * fraccion)
    barra = "█" * relleno + "░" * (longitud - relleno)
    return f"[{barra}] {fraccion * 100:5.1f}%"

def main():
    print("\n" + "=" * 78)
    print("📊 ESTADO DEL DATASET DE RESIDUOS (META: 1,000 FOTOGRAFÍAS LIMPIAS POR CLASE)")
    print("=" * 78)
    print(f"{'Categoría':<22} | {'Limpias':<8} | {'Descartes':<10} | {'Progreso (Meta: 1000)':<25}")
    print("-" * 78)

    total_limpias = 0
    total_descartes = 0

    for clase in CLASES:
        dir_val = os.path.join(CARPETA_DATASET, clase)
        dir_desc = os.path.join(CARPETA_DESCARTES, clase)

        n_val = contar_archivos(dir_val)
        n_desc = len(os.listdir(dir_desc)) if os.path.exists(dir_desc) else 0

        total_limpias += n_val
        total_descartes += n_desc

        p_bar = barra_progreso(n_val, META, 15)
        print(f"{clase:<22} | {n_val:<8} | {n_desc:<10} | {p_bar}")

    print("-" * 78)
    progreso_total = barra_progreso(total_limpias, len(CLASES) * META, 20)
    print(f"{'TOTAL GENERAL':<22} | {total_limpias:<8} | {total_descartes:<10} | {progreso_total}")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
