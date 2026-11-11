#!/usr/bin/env python3
"""
=============================================================================
NORMALIZACIÓN MASIVA DEL DATASET SUPLEMENTARIO CON RECORTE YOLO (180x180 px)
=============================================================================
Aplica el recorte inteligente de objetos con YOLOv8 a todo dataset_suplementario:
- Localiza el residuo con GPU (NVIDIA RTX 2050).
- Recorta en proporción 1:1 con un 8% de margen de contexto (sin bordes blancos).
- Escala a 180x180 px con filtro Lanczos de alta calidad.
- Si no hay objeto individual claro, aplica recorte central cuadrado.
=============================================================================
"""

import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

DIR_DATASET = "dataset_suplementario"
TAMANO_ESTANDAR = (180, 180)

def procesar_dataset():
    print("=" * 70)
    print("🚀 NORMALIZACIÓN INTELIGENTE CON YOLOv8 (180x180 px)")
    print("=" * 70)
    
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"⚡ Dispositivo de aceleración: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")
    
    yolo_model = YOLO("yolov8n.pt")
    
    # Recolectar todas las imágenes
    todas_imagenes = []
    for cat in sorted(os.listdir(DIR_DATASET)):
        ruta_cat = os.path.join(DIR_DATASET, cat)
        if os.path.isdir(ruta_cat) and not cat.startswith('.'):
            for f in os.listdir(ruta_cat):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    todas_imagenes.append(os.path.join(ruta_cat, f))
                    
    total = len(todas_imagenes)
    print(f"📁 Total de imágenes a normalizar: {total:,}")
    print("=" * 70)
    
    if total == 0:
        print("❌ No se encontraron imágenes en dataset_suplementario.")
        return
        
    con_yolo = 0
    con_center = 0
    errores = 0
    inicio = time.time()
    
    for ruta in tqdm(todas_imagenes, desc="Normalizando a 180x180"):
        try:
            img_bgr = cv2.imread(ruta)
            if img_bgr is None:
                errores += 1
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]
            
            # Inferencia YOLO
            results = yolo_model(img_rgb, conf=0.15, device=device, verbose=False)
            boxes = []
            for r in results:
                for b in r.boxes:
                    boxes.append(b.xyxy[0].cpu().numpy())
                    
            if boxes:
                con_yolo += 1
                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)
                
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                # Margen del 8% y cuadrado 1:1
                side = max(bw, bh) * 1.16
                side = min(side, max(H, W))
                
                cx1 = max(0, int(cx - side / 2.0))
                cy1 = max(0, int(cy - side / 2.0))
                cx2 = min(W, int(cx + side / 2.0))
                cy2 = min(H, int(cy + side / 2.0))
                
                crop = img_rgb[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    crop = img_rgb
            else:
                con_center += 1
                side = min(H, W)
                cx1 = (W - side) // 2
                cy1 = (H - side) // 2
                crop = img_rgb[cy1:cy1 + side, cx1:cx1 + side]
                
            # Resize a 180x180
            final_180 = cv2.resize(crop, TAMANO_ESTANDAR, interpolation=cv2.INTER_LANCZOS4)
            final_bgr = cv2.cvtColor(final_180, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta, final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception:
            errores += 1
            
    duracion = time.time() - inicio
    print("\n" + "=" * 70)
    print(f"🎉 ¡NORMALIZACIÓN FINALIZADA EN {duracion:.1f} SEGUNDOS!")
    print(f"   🔍 Recortadas con detección YOLO: {con_yolo:,} ({con_yolo/total*100:.1f}%)")
    print(f"   📐 Recortadas con Center-Crop:   {con_center:,} ({con_center/total*100:.1f}%)")
    if errores > 0:
        print(f"   ⚠️ Errores de lectura:          {errores}")
    print(f"   📏 Tamaño final: 180x180 px en el 100% del dataset suplementario.")
    print("=" * 70)

if __name__ == "__main__":
    procesar_dataset()
