#!/usr/bin/env python3
"""
Script de enriquecimiento "In-The-Wild" para las 9 clases base de RealWaste.
Elimina el sesgo de dominio (Domain Shift) entre las 9 clases de estudio (RealWaste)
y las 10 clases descargadas de la web, agregando fotografías reales caseras y variadas
de internet para las 9 categorías afectadas y des-sesgando el fondo gris de estudio.
"""
import os
import glob
import hashlib
import logging
import concurrent.futures
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

logging.getLogger('icrawler').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_completo")
RAW_WILD_DIR = os.path.join(BASE_DIR, "dataset_suplementario", "wild_9clases_raw")

QUERIES_9_CLASES = {
    "Cardboard": [
        "corrugated cardboard box waste real photo -vector -illustration",
        "cajas de carton reciclaje casero foto real",
        "flattened cardboard boxes pile household photo",
        "torn corrugated cardboard texture close up photo",
        "old shipping cardboard box trash photo",
        "carton corrugado pedazos basura foto"
    ],
    "Food_Organics": [
        "kitchen food scraps fruit vegetable peels waste photo",
        "residuos organicos comida cascaras frutas basura foto real",
        "banana peel apple core food waste kitchen counter photo",
        "leftover food waste compost bin household photo",
        "vegetable peelings organic waste pile photo",
        "restos de comida cascaras de verdura y huevo foto"
    ],
    "Glass": [
        "empty clear glass bottle recycling household photo -vector",
        "botellas de vidrio vacias reciclaje casero foto real",
        "green brown beer wine glass bottles waste photo",
        "empty glass jar container kitchen table photo",
        "discarded glass beverage bottles close up photo",
        "frascos y botellas de vidrio transparente basura foto"
    ],
    "Metal": [
        "crushed aluminum soda can waste photo -vector",
        "latas de aluminio y conservas reciclaje foto real",
        "empty tin food cans metal waste household photo",
        "discarded beverage cans aluminum scrap photo",
        "crushed beer soda cans on table photo",
        "lata de atun y refresco vacia metal basura foto"
    ],
    "Miscellaneous_Trash": [
        "black plastic garbage bag tied household trash photo",
        "bolsa de basura negra amarrada residuos ordinarios foto",
        "mixed non recyclable household waste wrappers dirty napkins photo",
        "dirty food wrappers chip bags trash bin photo",
        "full trash bag garbage kerbside photo",
        "basura ordinaria servilletas sucias envolturas bolsas foto"
    ],
    "Paper": [
        "crumpled white paper ball waste desk photo -vector",
        "papel arrugado hojas de papel reciclaje foto real",
        "old newspapers magazines paper stack waste photo",
        "printed office paper sheets trash bin photo",
        "torn notebook paper pages waste photo",
        "hojas de cuaderno y periodico viejo reciclaje foto"
    ],
    "Plastic": [
        "empty clear plastic water bottle crushed PET photo -vector",
        "botella de plastico PET transparente vacia reciclaje foto",
        "discarded plastic detergent shampoo bottles household photo",
        "disposable plastic cup bottle waste in hand photo",
        "plastic soda bottles recycling pile photo",
        "envases de plastico botellas pet basura casera foto"
    ],
    "Textile_Trash": [
        "old worn out clothes pile textile waste photo -vector",
        "ropa vieja usada trapos residuos textiles foto real",
        "discarded torn t-shirt jeans fabric rags waste photo",
        "used clothing pile thrift textile trash photo",
        "pile of old folded and crumpled clothes waste photo",
        "prendas de ropa rota retazos de tela basura foto"
    ],
    "Vegetation": [
        "fallen tree leaves dry and green garden waste photo",
        "hojas secas ramas jardineria poda residuos vegetales foto",
        "grass clippings yard waste pile photo",
        "pruned tree branches twigs leaves pile photo",
        "green plant leaves garden trimmings waste photo",
        "restos de poda hojas y ramas jardin basura foto"
    ]
}

def download_for_class(cls_name, queries, max_per_query=45):
    cls_raw_dir = os.path.join(RAW_WILD_DIR, cls_name)
    os.makedirs(cls_raw_dir, exist_ok=True)
    for i, q in enumerate(queries):
        sub_dir = os.path.join(cls_raw_dir, f"q_{i}")
        os.makedirs(sub_dir, exist_ok=True)
        existing = len(os.listdir(sub_dir))
        if existing >= max_per_query // 2:
            continue
        crawler = BingImageCrawler(
            downloader_threads=8,
            storage={'root_dir': sub_dir},
            log_level=logging.CRITICAL
        )
        crawler.crawl(keyword=q, max_num=max_per_query)

    # Procesar, validar y guardar como 180x180 en dataset_completo/{cls_name}
    target_dir = os.path.join(DATASET_DIR, cls_name)
    seen_hashes = set()
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp):
            try:
                with open(fp, 'rb') as fh:
                    seen_hashes.add(hashlib.md5(fh.read()).hexdigest())
            except Exception:
                pass

    added = 0
    for root, _, files in os.walk(cls_raw_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                img = Image.open(fp)
                if img.width < 80 or img.height < 80:
                    continue
                img_rgb = img.convert("RGB")
                arr = np.array(img_rgb)
                # Filtrar imagenes completamente blancas/negras o sin varianza
                if arr.std() < 18.0:
                    continue
                img_180 = img_rgb.resize((180, 180), Image.Resampling.BILINEAR)
                out_name = f"wild_{cls_name}_{added:04d}.jpg"
                out_path = os.path.join(target_dir, out_name)
                img_180.save(out_path, "JPEG", quality=92)
                with open(out_path, 'rb') as fh:
                    h = hashlib.md5(fh.read()).hexdigest()
                if h in seen_hashes:
                    os.remove(out_path)
                    continue
                seen_hashes.add(h)
                added += 1
            except Exception:
                continue
    print(f"✅ [{cls_name}] Nuevas imágenes wild añadidas: {added}")
    return cls_name, added

def harmonize_realwaste_studio_images():
    """
    Normaliza las imágenes RealWaste (524x524) a 180x180 y genera variantes con
    recorte central y fondos diversificados para romper el atajo del fondo gris-azulado
    de estudio [170, 180, 192].
    """
    rng = np.random.default_rng(42)
    studio_bg = np.array([170.0, 180.0, 192.0], dtype=np.float32)

    for cls_name in QUERIES_9_CLASES.keys():
        target_dir = os.path.join(DATASET_DIR, cls_name)
        orig_files = [
            f for f in os.listdir(target_dir)
            if not f.startswith("wild_") and not f.startswith("aug_debiased_") and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        debiased_count = 0
        for idx, fname in enumerate(orig_files):
            fpath = os.path.join(target_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                w, h = img.size
                # 1. Asegurar que la imagen original también esté a 180x180 como las otras 10 clases
                if (w, h) != (180, 180):
                    img_180 = img.resize((180, 180), Image.Resampling.BILINEAR)
                    img_180.save(fpath, "JPEG", quality=93)
                else:
                    img_180 = img

                # 2. Crear una versión des-sesgada (crop central + reemplazo de fondo de estudio) para el 50% de las muestras
                if idx % 2 == 0:
                    arr = np.array(img, dtype=np.float32)
                    # Recorte central del 76% para enfocarse en el objeto y quitar bordes de estudio
                    ch, cw = int(h * 0.76), int(w * 0.76)
                    y0, x0 = (h - ch) // 2, (w - cw) // 2
                    cropped = arr[y0:y0+ch, x0:x0+cw]
                    cropped_img = Image.fromarray(np.clip(cropped, 0, 255).astype(np.uint8)).resize((180, 180), Image.Resampling.BILINEAR)
                    c_arr = np.array(cropped_img, dtype=np.float32)

                    # Detectar píxeles de fondo de estudio y reemplazarlos por fondos variados (blanco, cálido, oscuro, natural)
                    dist = np.linalg.norm(c_arr - studio_bg, axis=-1)
                    mask_bg = dist < 34.0
                    mode = idx % 4
                    if mode == 0:
                        new_bg = np.full_like(c_arr, rng.uniform(235, 255)) # Fondo blanco/claro
                    elif mode == 1:
                        new_bg = np.empty_like(c_arr) # Fondo madera/mesa cálida
                        new_bg[..., 0] = rng.uniform(160, 210)
                        new_bg[..., 1] = rng.uniform(130, 175)
                        new_bg[..., 2] = rng.uniform(90, 135)
                    elif mode == 2:
                        new_bg = np.full_like(c_arr, rng.uniform(25, 70)) # Fondo oscuro/asfalto
                    else:
                        # Gradiente suave con ruido natural
                        new_bg = rng.uniform(80, 220, size=(1, 1, 3)) + rng.normal(0, 12, size=c_arr.shape)

                    c_arr[mask_bg] = new_bg[mask_bg]
                    out_debiased = os.path.join(target_dir, f"aug_debiased_{idx:04d}.jpg")
                    Image.fromarray(np.clip(c_arr, 0, 255).astype(np.uint8)).save(out_debiased, "JPEG", quality=91)
                    debiased_count += 1
            except Exception:
                continue
        print(f"🔧 [{cls_name}] Normalizadas a 180x180 y generadas {debiased_count} variantes sin fondo de estudio.")

if __name__ == "__main__":
    print("🚀 Iniciando descarga paralela de imágenes In-The-Wild para las 9 clases RealWaste...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_for_class, cls_name, queries)
            for cls_name, queries in QUERIES_9_CLASES.items()
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    print("🎨 Armonizando imágenes de estudio RealWaste y eliminando sesgo de fondo...")
    harmonize_realwaste_studio_images()
    print("✨ Enriquecimiento completado con éxito.")

