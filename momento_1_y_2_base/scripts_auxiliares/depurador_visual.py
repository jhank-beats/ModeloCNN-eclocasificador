#!/usr/bin/env python3
"""
=============================================================================
DEPURADOR VISUAL INTERACTIVO DE DATASET (GRADIO WEB APP)
=============================================================================
Permite inspeccionar y limpiar visualmente el dataset en tu navegador web.
- Haz clic en cualquier imagen para moverla a la papelera de descartes con 1 solo clic.
- Permite cambiar de categoría, paginar y ver estadísticas en vivo.
- Opción de recuperar la última imagen descartada por error.
=============================================================================
"""

import os
import shutil
import gradio as gr

DATASET_DIR = "dataset_suplementario"
PAPELERA_DIR = "dataset_suplementario_descartes"
PAGE_SIZE = 36  # 6 columnas x 6 filas

historial_descartes = []  # Para botón 'Deshacer último descarte'

def obtener_categorias():
    if not os.path.exists(DATASET_DIR):
        return []
    return sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')
    ])

def listar_archivos_categoria(categoria):
    if not categoria:
        return []
    ruta_cat = os.path.join(DATASET_DIR, categoria)
    if not os.path.exists(ruta_cat):
        return []
    return sorted([
        f for f in os.listdir(ruta_cat)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])

def cargar_galeria(categoria, pagina):
    if not categoria:
        return [], "⚠️ Selecciona una categoría", 1, "Página 1 de 1"
    
    archivos = listar_archivos_categoria(categoria)
    total = len(archivos)
    max_pags = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    pag_actual = max(1, min(pagina, max_pags))
    
    inicio = (pag_actual - 1) * PAGE_SIZE
    fin = inicio + PAGE_SIZE
    lote = archivos[inicio:fin]
    
    ruta_cat = os.path.join(DATASET_DIR, categoria)
    imagenes = [(os.path.join(ruta_cat, f), f) for f in lote]
    
    info_md = (
        f"### 📂 Categoría: **{categoria}**\n"
        f"📊 **{total}** imágenes limpias actualmente en esta clase | "
        f"📄 Página **{pag_actual}** de **{max_pags}**"
    )
    pag_txt = f"Pág {pag_actual}/{max_pags}"
    return imagenes, info_md, pag_actual, pag_txt

def eliminar_imagen(categoria, pagina, evt: gr.SelectData):
    global historial_descartes
    if not categoria:
        return "", [], "", pagina, ""
    
    # Extraer la ruta de la imagen seleccionada
    ruta = ""
    if isinstance(evt.value, dict):
        ruta = evt.value.get("image", {}).get("path", "")
    elif isinstance(evt.value, str):
        ruta = evt.value

    msg = ""
    if ruta and os.path.exists(ruta):
        nombre = os.path.basename(ruta)
        dest_carpeta = os.path.join(PAPELERA_DIR, categoria)
        os.makedirs(dest_carpeta, exist_ok=True)
        dest_archivo = os.path.join(dest_carpeta, f"manual_{nombre}")
        
        # Mover a descartes
        shutil.move(ruta, dest_archivo)
        historial_descartes.append((categoria, dest_archivo, ruta))
        msg = f"🗑️ Descartada: **{nombre}** (movida a descartes)"
    
    # Recargar la galería en la página actual
    imagenes, info_md, pag_actual, pag_txt = cargar_galeria(categoria, pagina)
    return msg, imagenes, info_md, pag_actual, pag_txt

def deshacer_ultimo(categoria, pagina):
    global historial_descartes
    if not historial_descartes:
        return "ℹ️ No hay descartes recientes para deshacer.", [], "", pagina, ""
    
    cat_ant, dest_archivo, ruta_original = historial_descartes.pop()
    msg = ""
    if os.path.exists(dest_archivo):
        shutil.move(dest_archivo, ruta_original)
        msg = f"↩️ Recuperada: **{os.path.basename(ruta_original)}** devuelta a **{cat_ant}**"
    
    imagenes, info_md, pag_actual, pag_txt = cargar_galeria(categoria, pagina)
    return msg, imagenes, info_md, pag_actual, pag_txt

def pagina_anterior(categoria, pagina):
    nueva_pag = max(1, pagina - 1)
    imgs, info, pag, txt = cargar_galeria(categoria, nueva_pag)
    return imgs, info, pag, txt, ""

def pagina_siguiente(categoria, pagina):
    imgs, info, pag, txt = cargar_galeria(categoria, pagina + 1)
    return imgs, info, pag, txt, ""

# =============================================================================
# CONSTRUCCIÓN DE LA INTERFAZ WEB
# =============================================================================
custom_css = """
.galeria-estilo { min-height: 600px; }
.btn-nav { font-weight: bold; }
"""

with gr.Blocks(title="🧹 Depurador Visual de Dataset - Residuos") as demo:
    gr.Markdown(
        """
        # 🧹 Depurador Visual Interactivo del Dataset
        ### Instrucciones:
        1. Selecciona la categoría que deseas revisar en el menú desplegable.
        2. **Haz clic directamente sobre cualquier imagen que NO sirva** (dibujo, afiche, texto, o contenido incorrecto) para moverla instantáneamente a la papelera de descartes.
        3. Si te equivocas, presiona **↩️ Deshacer último descarte** para restaurarla.
        """
    )
    
    cats = obtener_categorias()
    cat_inicial = cats[0] if cats else None
    
    with gr.Row():
        dropdown_cat = gr.Dropdown(
            choices=cats,
            value=cat_inicial,
            label="📁 Selecciona la Categoría a Depurar",
            scale=3
        )
        btn_recargar = gr.Button("🔄 Recargar", scale=1)
        btn_deshacer = gr.Button("↩️ Deshacer Último Descarte", variant="secondary", scale=2)

    info_status = gr.Markdown("Cargando categoría...")
    notif_msg = gr.Markdown("")
    
    with gr.Row():
        btn_prev = gr.Button("⬅️ Página Anterior", variant="primary", scale=1)
        label_pag = gr.Markdown("Pág 1", elem_id="pag_indicator")
        pag_state = gr.State(1)
        btn_next = gr.Button("Página Siguiente ➡️", variant="primary", scale=1)

    gallery = gr.Gallery(
        label="Fotografías de la clase (Haz clic en una imagen para descartarla con 1 clic)",
        columns=6,
        rows=6,
        height=700,
        object_fit="contain",
        elem_classes=["galeria-estilo"]
    )

    # Eventos interactivos
    dropdown_cat.change(
        fn=lambda c: cargar_galeria(c, 1) + ("",),
        inputs=[dropdown_cat],
        outputs=[gallery, info_status, pag_state, label_pag, notif_msg]
    )
    
    btn_recargar.click(
        fn=lambda c, p: cargar_galeria(c, p) + ("",),
        inputs=[dropdown_cat, pag_state],
        outputs=[gallery, info_status, pag_state, label_pag, notif_msg]
    )
    
    btn_prev.click(
        fn=pagina_anterior,
        inputs=[dropdown_cat, pag_state],
        outputs=[gallery, info_status, pag_state, label_pag, notif_msg]
    )
    
    btn_next.click(
        fn=pagina_siguiente,
        inputs=[dropdown_cat, pag_state],
        outputs=[gallery, info_status, pag_state, label_pag, notif_msg]
    )

    gallery.select(
        fn=eliminar_imagen,
        inputs=[dropdown_cat, pag_state],
        outputs=[notif_msg, gallery, info_status, pag_state, label_pag]
    )

    btn_deshacer.click(
        fn=deshacer_ultimo,
        inputs=[dropdown_cat, pag_state],
        outputs=[notif_msg, gallery, info_status, pag_state, label_pag]
    )

    # Carga inicial al abrir
    demo.load(
        fn=lambda: cargar_galeria(cat_inicial, 1) + ("",),
        outputs=[gallery, info_status, pag_state, label_pag, notif_msg]
    )

if __name__ == "__main__":
    for puerto in range(7862, 7880):
        try:
            demo.launch(server_name="127.0.0.1", server_port=puerto, css=custom_css)
            break
        except OSError:
            continue
