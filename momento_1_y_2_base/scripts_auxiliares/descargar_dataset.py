#!/usr/bin/env python3
"""
=============================================================================
SCRIPT PROFESIONAL DE DESCARGA Y FILTRADO AUTOMÁTICO DE DATASET (10 CLASES)
=============================================================================
Objetivo:
- Descargar exactamente 1,000 fotografías REALES, nítidas y sin texto por categoría.
- Filtro estricto: Descarta infografías, dibujos, vectores, cliparts, animaciones,
  imágenes borrosas, duplicados y fondos planos artificiales.
- Almacenamiento ordenado en 'dataset_suplementario' y descartes en
  'dataset_suplementario_descartes'.
- IDEMPOTENTE: Preserva todas las fotos previamente descargadas y no re-descarga
  duplicados.
=============================================================================
"""

import os
import sys
import time
import shutil
import logging
import json
import cv2
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

# Suprimir logs verbosos de bibliotecas
logging.getLogger('icrawler').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

OBJETIVO_POR_CLASE = 1000
CARPETA_DESTINO = "dataset_suplementario"
CARPETA_DESCARTES = "dataset_suplementario_descartes"
PROGRESS_FILE = "dataset_progreso.json"

# =============================================================================
# DEFINICIÓN DE CATEGORÍAS Y BÚSQUEDAS FOTOGRÁFICAS EXHAUSTIVAS (35+ POR CLASE)
# =============================================================================
CATEGORIAS = {
    "Pilas_Baterias": [
        "used aa aaa batteries pile photo -cartoon -drawing -vector",
        "discarded household batteries recycling container real photo",
        "pilas y baterias usadas contenedor reciclaje fotografia real",
        "old car battery scrap yard photo -diagram",
        "lithium ion batteries waste scrap photo",
        "baterias de celular y laptop desechadas foto real",
        "swollen lithium battery trash photo",
        "worn alkaline batteries pile close up photo",
        "pila de baterias gastadas reciclaje foto",
        "discarded button cell batteries coin cells photo",
        "baterias de plomo acido viejas desecho foto",
        "pile of dead batteries recycling facility",
        "battery waste bin collection photo",
        "baterias usadas chatarra reciclaje fotografia",
        "corroded leaky batteries waste photo",
        "rechargeable batteries recycling depot photo",
        "pilas boton sulfatadas desecho foto",
        "acumuladores de plomo desecho reciclaje foto",
        "pile of 9v batteries discarded photo",
        "e-waste battery collection bin photo",
        "industrial battery scrap disposal photo",
        "piles usees bac recyclage photo",
        "altbatterien sammeln entsorgung foto",
        "baterias gastas reciclagem foto",
        "lithium pouch cell damage waste photo",
        "motorcycle battery scrap pile photo",
        "pilas alcalinas en basurero foto",
        "dead batteries macro close up photo",
        "car battery recycling lead recovery photo",
        "discarded powerbank lithium battery scrap photo",
        "electric scooter battery pack waste photo",
        "used alkaline batteries sorting facility photo",
        "pila de baterias usadas en caja foto",
        "batterie scariche raccolta differenziata foto",
        "worn out rechargeable battery trash photo"
    ],
    "RAEE_Electronicos": [
        "electronic waste e-waste circuit boards photo -vector -diagram",
        "discarded broken computers electronics scrap photo",
        "basura electronica raee placas cables fotografia real",
        "scrap motherboards pile e-waste photo",
        "broken laptops discarded trash pile photo",
        "cables y placas electronicas desechadas foto real",
        "discarded computer parts e-waste facility photo",
        "old broken cathode ray monitors trash photo",
        "chatarra electronica componentes reciclaje foto",
        "discarded cell phones pile e-waste photo",
        "discarded printers small appliances scrap photo",
        "desechos electronicos reciclaje taller foto",
        "scrap computer towers open e-waste photo",
        "pile of power adapters chargers wires trash",
        "broken flat screen tv trash photo",
        "desguace de electrodomesticos chatarra foto",
        "printed circuit board scrap shredding photo",
        "disipadores y ventiladores pc desecho foto",
        "broken microwave toaster appliance trash photo",
        "electronic components scrap sorting center",
        "viejo teclado y raton ordenador basura foto",
        "e-waste recycling plant workers heap photo",
        "elektronikschrott platine kabel entsorgung foto",
        "dechets electroniques carte mere photo",
        "lixo eletronico placas circuitos foto",
        "old hard drives disassembled scrap photo",
        "modem router cables pile trash photo",
        "broken gaming consoles scrap electronics photo",
        "chatarra de cobre y cables electricos desecho",
        "industrial control board scrap ewaste photo",
        "broken tablets e-waste heap photo",
        "raee residuos electronicos contenedor limpio foto",
        "salvaged electronic scrap pile workshop photo",
        "discarded audio equipment amplifier scrap photo",
        "server rack scrap ewaste photo"
    ],
    "Medicamentos": [
        "expired medicine blister packs pills waste photo -vector -infographic",
        "discarded prescription bottles pills trash photo",
        "medicamentos vencidos cajas pastillas desecho fotografia real",
        "pharmaceutical waste pills blisters disposal photo",
        "blister de pastillas vacios y vencidos foto real",
        "expired medical syrups bottles waste photo",
        "ampollas y frascos de medicamentos desecho foto",
        "discarded medication blister packs trash photo",
        "punto sigre medicamentos residuos foto real",
        "waste pill bottles pharmacy disposal photo",
        "descarte de medicamentos ampollas pastillas foto",
        "empty and expired medicine packs disposal photo",
        "pile of empty pill blisters recycling photo",
        "discarded glass vials medicine trash photo",
        "capsulas y pastillas caducadas contenedor foto",
        "pharmaceutical waste bin hospital photo",
        "expired antibiotics blister packs photo",
        "frascos de jarabe vacios desechados foto",
        "medical blister packaging scrap photo",
        "pomadas y cremas medicinales vencidas foto",
        "medicaments perimes poubelle tri photo",
        "abgelaufene medikamente entsorgung foto",
        "medicamentos fora da validade lixo foto",
        "unwanted medicines collection pharmacy photo",
        "expired eye drops inhaler medical waste photo",
        "empty foil pill strips waste photo",
        "sobrantes de medicinas desecho foto",
        "pharmaceutical disposal incinerator bin photo",
        "crushed expired pills disposal photo",
        "pastillas sueltas caducadas en basurero foto",
        "medical blister pack crushing recycling photo",
        "discarded insulin pens empty vials photo",
        "cajas de remedios vencidos basurero foto",
        "medicinali scaduti raccolta differenziata foto",
        "expired pharmaceutical supplies disposal photo"
    ],
    "Madera_Procesada": [
        "scrap wood pallets broken lumber waste photo -drawing",
        "broken wooden pallets pile scrap yard photo",
        "madera pallets tablas rotas desecho fotografia real",
        "carpentry scrap timber offcuts pile photo",
        "muebles de madera rotos escombros desecho foto",
        "discarded particle board mdf pieces scrap photo",
        "restos de madera procesada aserradero foto real",
        "construction timber scrap woodpile photo",
        "tableros de madera contrachapada desecho foto",
        "broken wood crates scrap waste photo",
        "demolition timber wood scrap photo",
        "listones y tablas de madera desechadas foto",
        "scrap plywood cutoffs carpentry workshop photo",
        "desechos de melamina madera aglomerada foto",
        "broken wooden chair table furniture trash photo",
        "timber offcuts waste bin carpentry photo",
        "palets rotos astillados vertedero foto",
        "wood recycling shredder waste pile photo",
        "vigas y tablas de encofrado usadas foto",
        "scaffold boards old timber scrap photo",
        "dechets de bois menuiserie palettes photo",
        "altholz sperrmull paletten entsorgung foto",
        "madeira de demolicao restos sucata foto",
        "carpentry sawdust and wood offcuts pile photo",
        "broken wooden doors planks discarded photo",
        "industrial wooden packaging waste photo",
        "scrap lumber pile construction dumpster photo",
        "mueble aglomerado roto basura calle foto",
        "cajas de fruta madera rotas desecho foto",
        "treated lumber scrap disposal site photo",
        "weathered pallet wood scrap pile photo",
        "wood chip biomass timber waste photo",
        "tablas de pino cepilladas desecho taller foto",
        "scarti di legno falegnameria foto",
        "broken bed frame wooden slats trash photo"
    ],
    "Escombros_Ceramica": [
        "demolition rubble broken bricks concrete ceramic photo -vector",
        "broken ceramic tiles rubble scrap photo",
        "escombros construccion ladrillos ceramica azulejos foto real",
        "construction masonry rubble concrete chunks photo",
        "restos de azulejos baldosas rotas escombros foto",
        "demolished brick wall rubble pile photo",
        "trozos de hormigon ceramica escombros demolicion foto",
        "cracked porcelain tiles scrap construction waste photo",
        "escombros de reforma albañileria ladrillo foto real",
        "broken bathroom ceramic toilet sink demolition photo",
        "concrete and brick debris construction site photo",
        "residuos de ceramica ladrillos obra foto",
        "broken ceramic roof tiles scrap pile photo",
        "desechos de yeso escayola escombros foto",
        "shattered porcelain tiles bucket renovation photo",
        "crushed concrete gravel demolition recycling photo",
        "demolition skip rubble bricks tiles photo",
        "lavabo y bidet de porcelana roto escombros foto",
        "broken terracotta floor tiles scrap photo",
        "mortero y pedazos de pared caida foto",
        "gravats de demolition briques carrelage photo",
        "bauschutt abbruch ziegel fliesen foto",
        "entulho de construcao tijolos ceramica foto",
        "cracked mortar concrete slabs demolition photo",
        "ceramic sanitaryware demolition scrap photo",
        "pile of broken masonry blocks skips photo",
        "trozos de baldosas y cemento escombros foto",
        "rubble pile building site restoration photo",
        "broken ceramic pipes clay tile scrap photo",
        "azulejos arrancados de pared desecho foto",
        "crushed brick aggregate demolition waste photo",
        "demolished chimney bricks mortar rubble photo",
        "macerie edili mattoni piastrelle foto",
        "excavator loading construction rubble bricks photo",
        "remodel debris broken tiles drywall photo"
    ],
    "Bombillas": [
        "burned light bulbs fluorescent tubes waste photo -vector",
        "broken incandescent light bulbs scrap photo",
        "bombillas focos fluorescentes fundidos desecho foto real",
        "compact fluorescent lamps cfl waste container photo",
        "tubos fluorescentes rotos y fundidos reciclaje foto",
        "discarded led and halogen light bulbs trash photo",
        "focos quemados desecho reciclaje foto real",
        "burned out light bulbs pile recycling bin photo",
        "lamparas y bombillas usadas contenedor reciclaje foto",
        "pile of old bulbs waste sorting photo",
        "bombillas incandescentes viejas rotas foto",
        "fluorescent light tube waste collection photo",
        "broken glass lightbulb recycling box photo",
        "focos ahorradores espiral fundidos desecho foto",
        "tubos t8 fluorescentes usados caja reciclaje foto",
        "discarded ceiling light fixtures bulbs photo",
        "halogen spotlight bulbs burned out scrap photo",
        "ampoules usees recyclage dechetterie photo",
        "altlampen gluhbirnen leuchtstoffrohren foto",
        "lampadas queimadas reciclagem foto",
        "lampadine fulminate raccolta differenziata foto",
        "mercury vapor lamp hazardous waste photo",
        "pile of burned edison light bulbs trash photo",
        "broken led light bulbs plastic diffusor photo",
        "focos led danados chatarra reciclaje foto",
        "fluorescent tube crushing recycling machine photo",
        "pile of cfl spiral lamps disposal bin photo",
        "bombillas dicroicas halogenas fundidas foto",
        "commercial light fixture scrap lamps photo",
        "used street lamp bulb sodium discharge photo",
        "burned car headlight bulbs scrap photo",
        "focos de filamento rotos contenedor foto",
        "lamp recycling collection center boxes photo",
        "discarded chandelier light bulbs pile photo",
        "fluorescent ballast and tube waste photo"
    ],
    "Neumaticos_Gomas": [
        "discarded car tires pile rubber waste scrap photo -drawing",
        "old automobile tyres scrap yard photo",
        "neumaticos usados llantas desecho reciclaje fotografia real",
        "worn truck tires pile landfill rubber photo",
        "llantas de carro viejas apiladas foto real",
        "shredded scrap tires rubber waste photo",
        "neumaticos viejos vertedero reciclaje foto",
        "pile of used vehicle tires recycling plant photo",
        "caucho neumaticos desgastados desecho foto",
        "scrap tractor and car tires pile photo",
        "cubiertas de autos usadas reciclaje foto",
        "rubber tires waste dump site photo",
        "bald tread car tire wire showing scrap photo",
        "punctured damaged tire rim scrap photo",
        "pile of bicycle motorcycle tires trash photo",
        "ruedas usadas desguace apiladas foto",
        "shredded tyre crumb rubber granules photo",
        "pneus usages decharge recyclage photo",
        "altreifen gummischrott recycling foto",
        "pneus velhos sucata reciclagem foto",
        "pneumatici usati discarica foto",
        "heavy machinery earthmover tires scrap photo",
        "industrial conveyor belt rubber scrap pile photo",
        "llantas de camion gastadas desecho foto",
        "tire shredding facility conveyor photo",
        "discarded inner tubes rubber waste photo",
        "gomas y mangueras industriales desecho foto",
        "old forklift solid tires scrap photo",
        "abandoned tires ditch roadside dump photo",
        "pile of used winter tires disposal photo",
        "vulcanized rubber scrap pieces workshop photo",
        "cubiertas recauchutadas reventadas foto",
        "scrap retread tire casings pile photo",
        "used tires container collection facility photo",
        "mountain of old automobile tires dump photo"
    ],
    "Aceites_Quimicos": [
        "used motor oil containers chemical bottles waste photo -vector",
        "toxic chemical jugs discarded oil canisters photo",
        "envases de aceite lubricante quimicos industriales foto real",
        "discarded automotive fluid bottles trash photo",
        "garrafas de quimicos vacias desechadas foto",
        "waste industrial lubricant containers photo",
        "bidones y botellas de aceite usado mecanica foto real",
        "hazardous chemical plastic bottles waste photo",
        "envases de anticongelante aceite de motor desecho foto",
        "discarded dirty oil filters and jugs photo",
        "recipientes de quimicos y disolventes desecho foto",
        "empty pesticide chemical containers waste photo",
        "used oil drain pan container workshop photo",
        "plastic jerrycans chemical waste disposal photo",
        "garrafas de liquido de frenos y aceite vacias foto",
        "industrial solvent drums empty scrap photo",
        "bidones de pintura y disolvente vacios taller foto",
        "coolant antifreeze jugs empty in trash photo",
        "petrochemical plastic bottles waste photo",
        "envases plasticos de reactivos quimicos desecho foto",
        "bidons huile moteur usagee dechetterie photo",
        "altoldosen chemikalienkanister entsorgung foto",
        "frascos de oleo automotivo sucata foto",
        "fustini olio motore vuoti officina foto",
        "grease cartridges used automotive waste photo",
        "car cleaning chemical bottles discarded photo",
        "empty acid bleach plastic jugs waste photo",
        "garrafas aceite de transmission desecho foto",
        "hazardous waste collection drums chemical photo",
        "oily plastic bottles recycling facility photo",
        "detergente industrial bidones vacios foto",
        "dirty lubricant containers scrap heap photo",
        "residuos peligrosos envases hidrocarburos foto",
        "empty resin and hardener chemical cans photo",
        "discarded brake fluid bottles garage photo"
    ],
    "Residuos_Sanitarios": [
        "discarded medical surgical masks gloves waste photo -vector",
        "used biohazard medical waste yellow bag trash photo",
        "mascarillas guantes desechados basura hospitalaria foto real",
        "discarded medical gloves cotton gauze waste photo",
        "residuos sanitarios gasas guantes usados foto real",
        "used medical syringes in disposal bin photo",
        "bolsas rojas y amarillas residuos hospitalarios foto",
        "discarded protective medical suits trash photo",
        "desechos biologicos clinica hospital contenedor foto",
        "used disposable medical bandages gauze photo",
        "contenedor de residuos sanitarios desechables foto",
        "medical clinic trash bin gloves masks photo",
        "pile of blue disposable surgical masks trash photo",
        "used latex nitrile gloves discarded in bin photo",
        "sharps container full needles syringes photo",
        "apósitos gasas con sangre contenedor clinico foto",
        "discarded iv drip bags tubing hospital trash photo",
        "tubos de ensayo y cateteres desecho medico foto",
        "yellow clinical waste bags disposal site photo",
        "batas quirurgicas desechables basura foto",
        "dechets medicaux masques seringues poubelle photo",
        "klinischer abfall spritzen mundschutz foto",
        "lixo hospitalar luvas mascaras seringas foto",
        "rifiuti sanitari ospedalieri guanti foto",
        "used covid antigen rapid test kits trash photo",
        "medical cotton swabs disposal bin photo",
        "discarded speculum disposable medical plastic photo",
        "hospital disinfection wipes trash pile photo",
        "contenedor de agujas usadas biopeligro foto",
        "dental clinic disposable waste bibs cotton photo",
        "surgical suction canisters waste disposal photo",
        "used medical face shields disposable trash photo",
        "plastic medicine measuring cups waste photo",
        "hospital bed pads chux disposable trash photo",
        "sanitary waste container clinical sorting photo"
    ],
    "Tetra_Pack": [
        "envase tetra brik de leche foto -cardboard -shipping -caja -mudanza",
        "brick de leche vacio en la basura foto -caja -amazon",
        "brik de leche aplastado reciclaje foto -cardboard -box",
        "tetrabrik de leche desnatada residuo foto -caja",
        "brick de bebida vegetal avena soja residuo foto -caja",
        "briks de leche vacios contenedor foto -cardboard",
        "tetra brik leche tirado basura foto -caja",
        "brick de zumo vacio residuo foto -cardboard -box",
        "tetrabrik de zumo de naranja foto -caja",
        "brik de zumo aplastado basura foto -cardboard",
        "tetrabrik de batido vacio residuo foto",
        "envase tetra brik de zumo reciclaje foto -caja",
        "tetrabrik de caldo de pollo foto -caja",
        "brick de tomate frito vacio foto -caja",
        "tetrabrik de vino don simon foto -caja",
        "brick de vino basura residuo foto -caja",
        "brik de nata para cocinar residuo foto -caja",
        "envases tetra brik contenedor amarillo foto -cardboard",
        "montoncito de briks de leche reciclaje foto -caja",
        "briks de leche aplastados punto limpio foto -caja",
        "residuos envases brik tetrapak foto -cardboard",
        "envase tetra brik tirado en la calle foto -caja",
        "tetra brik aplastado para reciclar foto -caja",
        "envases tetra pak aplastados residuo foto -cardboard",
        "tetra prisma envase bebida vacio foto -box",
        "brique de lait tetra pak dechet photo -carton -box",
        "brique de lait ecrasee recyclage photo -carton",
        "brique de jus de fruit tetra pak poubelle photo -box",
        "briques alimentaires dechets tri photo -carton",
        "emballage tetra brik dechet photo -carton",
        "getrankekarton tetra pak muell foto -karton",
        "leere milchtuete tetra pak muell foto -kiste",
        "getrankekarton tetra brik gelbe tonne foto",
        "caixa de leite longa vida reciclagem foto -papelao",
        "embalagem longa vida tetra pak reciclagem foto -papelao",
        "caixinha de leite tetra pak descarte foto -papelao",
        "embalagens tetra pak reciclagem cooperativa foto",
        "confezione tetrapak latte riciclo foto -scatola",
        "cartoni tetrapak schiacciati raccolta differenziata foto",
        "flattened tetra pak milk carton recycling photo -corrugated -shipping",
        "crushed tetra pak beverage carton trash photo -box",
        "empty tetra pak juice carton trash photo -box",
        "tetra pak aseptic cartons waste photo -shipping",
        "squashed tetra brik beverage carton photo -box"
    ]
}

# =============================================================================
# ALGORITMO DE DETECCIÓN Y FILTRADO DE CALIDAD ESTRICTO
# =============================================================================
def calcular_dhash(img_bgr, hash_size=8):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    peq = cv2.resize(gris, (hash_size + 1, hash_size))
    diff = peq[:, 1:] > peq[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def distancia_hamming(h1, h2):
    return bin(h1 ^ h2).count('1')

def evaluar_imagen(ruta_archivo):
    try:
        with Image.open(ruta_archivo) as pil_img:
            pil_img.verify()
        with Image.open(ruta_archivo) as pil_img:
            if pil_img.format not in ['JPEG', 'PNG', 'WEBP', 'BMP']:
                return False, "formato_no_soportado", None, 0
    except Exception:
        return False, "corrupta_pil", None, 0

    img_bgr = cv2.imread(ruta_archivo)
    if img_bgr is None:
        return False, "corrupta_opencv", None, 0

    h, w = img_bgr.shape[:2]
    if h < 200 or w < 200:
        return False, "resolucion_baja", None, 0

    aspect = w / float(h)
    if aspect < 0.35 or aspect > 2.85:
        return False, "aspect_ratio_extremo", None, 0

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brillo = np.mean(gris)
    desv = np.std(gris)
    if brillo < 20 or (brillo < 30 and desv < 15):
        return False, "demasiado_oscura", None, 0
    if brillo > 240 and desv < 15:
        return False, "quemada_blanca", None, 0
    if desv < 18:
        return False, "sin_contraste", None, 0

    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
    if lap_var < 15.0:
        return False, "borrosa_o_plana", None, 0
    if lap_var > 4500.0 and desv < 35:
        return False, "grafico_alto_contraste", None, 0

    # Detección de fondo plano (infografía / dibujo)
    _, counts = np.unique(gris, return_counts=True)
    max_freq = np.max(counts) / float(gris.size)
    if max_freq > 0.40:
        return False, f"fondo_plano_infografia_{max_freq:.2f}", None, 0

    # Detección de dibujos por paleta cuantizada reducida
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_peq = cv2.resize(img_rgb, (64, 64))
    img_cuant = (img_peq // 32) * 32
    colores_unicos = len(np.unique(img_cuant.reshape(-1, 3), axis=0))
    if colores_unicos < 20:
        return False, f"pocos_colores_dibujo_{colores_unicos}", None, 0

    # Detección de texto denso / afiches
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = (edge_mag > 50).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    text_morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    text_ratio = np.count_nonzero(text_morph) / float(text_morph.size)
    if text_ratio > 0.45:
        return False, f"demasiado_texto_{text_ratio:.2f}", None, 0

    dhash = calcular_dhash(img_bgr)
    return True, "OK", img_bgr, dhash

# =============================================================================
# GESTIÓN DE PROGRESO Y DESCARGA
# =============================================================================
def contar_imagenes_validas(carpeta):
    if not os.path.exists(carpeta):
        return 0
    return len([f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])

def procesar_lote_temporal(carpeta_temp, carpeta_destino, carpeta_descartes, hashes_existentes):
    os.makedirs(carpeta_destino, exist_ok=True)
    os.makedirs(carpeta_descartes, exist_ok=True)

    archivos_temp = sorted([
        os.path.join(carpeta_temp, f) for f in os.listdir(carpeta_temp)
        if os.path.isfile(os.path.join(carpeta_temp, f)) and not f.startswith('.')
    ])

    validadas = 0
    descartadas = 0

    archivos_dest = [f for f in os.listdir(carpeta_destino) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    idx_actual = len(archivos_dest) + 1

    for ruta in archivos_temp:
        es_valida, razon, img_bgr, dhash = evaluar_imagen(ruta)
        nombre_base = os.path.basename(ruta)

        if es_valida:
            es_duplicada = False
            for h in hashes_existentes:
                if distancia_hamming(dhash, h) <= 2:
                    es_duplicada = True
                    break

            if es_duplicada:
                dest_desc = os.path.join(carpeta_descartes, f"duplicada_{nombre_base}")
                try:
                    shutil.move(ruta, dest_desc)
                except Exception:
                    pass
                descartadas += 1
            else:
                hashes_existentes.add(dhash)
                nombre_nuevo = f"{idx_actual:06d}.jpg"
                dest_valido = os.path.join(carpeta_destino, nombre_nuevo)
                cv2.imwrite(dest_valido, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                idx_actual += 1
                validadas += 1
                try:
                    os.remove(ruta)
                except Exception:
                    pass
        else:
            dest_desc = os.path.join(carpeta_descartes, f"{razon}_{nombre_base}")
            try:
                shutil.move(ruta, dest_desc)
            except Exception:
                pass
            descartadas += 1

    return validadas, descartadas

def descargar_categoria(nombre_clase, queries, objetivo=OBJETIVO_POR_CLASE):
    destino_cat = os.path.join(CARPETA_DESTINO, nombre_clase)
    descartes_cat = os.path.join(CARPETA_DESCARTES, nombre_clase)
    temp_cat = os.path.join(CARPETA_DESTINO, f"_temp_{nombre_clase}")
    os.makedirs(destino_cat, exist_ok=True)
    os.makedirs(descartes_cat, exist_ok=True)

    actuales = contar_imagenes_validas(destino_cat)
    print(f"\n=======================================================")
    print(f"📦 INICIANDO: [{nombre_clase}]")
    print(f"   Estado inicial: {actuales}/{objetivo} imágenes limpias")
    print(f"=======================================================")

    if actuales >= objetivo:
        print(f"✅ [{nombre_clase}] ya tiene {actuales} imágenes limpias (Meta cumplida).")
        return actuales

    hashes_existentes = set()
    for f in os.listdir(destino_cat):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            ruta_f = os.path.join(destino_cat, f)
            im = cv2.imread(ruta_f)
            if im is not None:
                hashes_existentes.add(calcular_dhash(im))

    for idx_q, query in enumerate(queries, 1):
        actuales = contar_imagenes_validas(destino_cat)
        if actuales >= objetivo:
            break

        faltantes = objetivo - actuales
        print(f"\n🔎 [{nombre_clase}] Query {idx_q}/{len(queries)}: '{query}'")
        print(f"   Faltan: {faltantes} | Actuales: {actuales}/{objetivo}")

        if os.path.exists(temp_cat):
            shutil.rmtree(temp_cat, ignore_errors=True)
        os.makedirs(temp_cat, exist_ok=True)

        cantidad_a_pedir = min(faltantes + 50, 150)

        try:
            crawler = BingImageCrawler(
                storage={'root_dir': temp_cat},
                downloader_threads=16,
                feeder_threads=2,
                parser_threads=4,
                log_level=logging.CRITICAL
            )
            crawler.crawl(
                keyword=query,
                filters=dict(type='photo', color='color'),
                min_size=(220, 220),
                max_num=cantidad_a_pedir,
                file_idx_offset='auto'
            )
        except Exception as e:
            print(f"   ⚠️ Excepción en crawler: {e}")

        val, desc = procesar_lote_temporal(temp_cat, destino_cat, descartes_cat, hashes_existentes)
        actuales = contar_imagenes_validas(destino_cat)
        print(f"   📊 Lote procesado: +{val} válidas agregadas | -{desc} descartadas")
        print(f"   🎯 Total acumulado en [{nombre_clase}]: {actuales}/{objetivo}")
        time.sleep(2)

    # Si aún no se alcanza la meta de 1,000, recurrir a combinaciones extendidas
    if actuales < objetivo:
        print(f"\n🔄 [{nombre_clase}] Generando consultas extendidas especializadas para completar {objetivo} fotos...")
        termino_base = nombre_clase.replace('_', ' ')
        variaciones_extra = [
            f"{termino_base} waste landfill site photo",
            f"{termino_base} recycling sorting plant photo",
            f"{termino_base} street litter trash photo",
            f"{termino_base} scrap yard pile real photo",
            f"{termino_base} broken damaged discarded photo",
            f"{termino_base} municipal waste disposal bin photo",
            f"{termino_base} dechet recyclage poubelle",
            f"{termino_base} entsorgung recycling abfall foto",
            f"{termino_base} residuo basura vertedero foto",
            f"{termino_base} contenedor de recogida reciclaje foto",
            f"{termino_base} chatarra desecho industrial foto",
            f"{termino_base} decharge publique dechets photo",
            f"{termino_base} close up real texture waste photo",
            f"{termino_base} collection point containers photo"
        ]

        for idx_ext, query_ext in enumerate(variaciones_extra, 1):
            actuales = contar_imagenes_validas(destino_cat)
            if actuales >= objetivo:
                break
            faltantes = objetivo - actuales
            print(f"\n🔎 [{nombre_clase}] Consulta Extendida {idx_ext}: '{query_ext}'")
            print(f"   Faltan: {faltantes} | Actuales: {actuales}/{objetivo}")

            if os.path.exists(temp_cat):
                shutil.rmtree(temp_cat, ignore_errors=True)
            os.makedirs(temp_cat, exist_ok=True)

            try:
                crawler = BingImageCrawler(
                    storage={'root_dir': temp_cat},
                    downloader_threads=16,
                    feeder_threads=2,
                    parser_threads=4,
                    log_level=logging.CRITICAL
                )
                crawler.crawl(
                    keyword=f"{query_ext} -cartoon -drawing -vector -clipart -diagram -infographic",
                    filters=dict(type='photo', color='color'),
                    min_size=(220, 220),
                    max_num=min(faltantes + 50, 150),
                    file_idx_offset='auto'
                )
            except Exception as e:
                print(f"   ⚠️ Excepción en crawler extendido: {e}")

            val, desc = procesar_lote_temporal(temp_cat, destino_cat, descartes_cat, hashes_existentes)
            actuales = contar_imagenes_validas(destino_cat)
            print(f"   📊 Lote extendido: +{val} válidas | -{desc} descartadas")
            print(f"   🎯 Total acumulado en [{nombre_clase}]: {actuales}/{objetivo}")
            time.sleep(2)

    if os.path.exists(temp_cat):
        shutil.rmtree(temp_cat, ignore_errors=True)

    print(f"\n🏁 [{nombre_clase}] Finalizado con {actuales}/{objetivo} imágenes.")
    return actuales

def main():
    print("=" * 70)
    print("🚀 PIPELINE DE DESCARGA Y LIMPIEZA MASIVA DE RESIDUOS (10 CLASES)")
    print(f"🎯 Meta: {OBJETIVO_POR_CLASE} fotografías reales por categoría (Total 10,000)")
    print(f"📁 Directorio de salida: '{CARPETA_DESTINO}'")
    print(f"🗑️ Directorio de descartes: '{CARPETA_DESCARTES}'")
    print("=" * 70)

    os.makedirs(CARPETA_DESTINO, exist_ok=True)
    os.makedirs(CARPETA_DESCARTES, exist_ok=True)

    totales = {}
    inicio_general = time.time()

    for clase, queries in CATEGORIAS.items():
        total_clase = descargar_categoria(clase, queries, OBJETIVO_POR_CLASE)
        totales[clase] = total_clase

        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                "timestamp": time.ctime(),
                "progreso": totales,
                "total_general": sum(totales.values()),
                "meta_general": len(CATEGORIAS) * OBJETIVO_POR_CLASE
            }, f, indent=2)

    tiempo_min = (time.time() - inicio_general) / 60
    print("\n" + "=" * 70)
    print("🎉 DESCARGA Y LIMPIEZA COMPLETADA CON ÉXITO")
    print(f"⏱️ Tiempo total: {tiempo_min:.1f} minutos")
    print("Resumen por categoría:")
    for cat, n in totales.items():
        estado = "✅ COMPLETA" if n >= OBJETIVO_POR_CLASE else f"⚠️ {n}/{OBJETIVO_POR_CLASE}"
        print(f" - {cat:22s}: {n:4d} fotos limpias [{estado}]")
    print(f"Total general de fotografías limpias: {sum(totales.values())}")
    print("=" * 70)

if __name__ == "__main__":
    main()
