# -*- coding: utf-8 -*-
"""Genera visualización de MixUp y CutMix para la diapositiva 13."""
import cv2, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sharp(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(g, cv2.CV_64F).var() if g is not None else -1

def best(pat):
    bf, bs = None, -1
    for f in glob.glob(pat):
        s = sharp(f)
        if s > bs: bf, bs = f, s
    return bf

A_path = best("dataset_completo/Cardboard/*.jpg")
B_path = best("dataset_completo/Metal/*.jpg")

A = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB).astype(np.float32)
B = cv2.cvtColor(cv2.imread(B_path), cv2.COLOR_BGR2RGB).astype(np.float32)
A = cv2.resize(A, (180, 180)); B = cv2.resize(B, (180, 180))

# --- MixUp (lambda = 0.5) ---
lam = 0.5
mixup = lam * A + (1.0 - lam) * B

# --- CutMix (parche de B pegado sobre A) ---
A_cut = A.copy()
H, W = 180, 180
ry, rx = 80, 80           # tamaño del parche
cx, cy = 50, 50           # esquina superior-izquierda del parche
A_cut[cy:cy+ry, cx:cx+rx] = B[cy:cy+ry, cx:cx+rx]
cutmix = A_cut

# --- Composición 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
panels = [
    (axes[0,0], A,       "Imagen A (Cartón)"),
    (axes[0,1], B,       "Imagen B (Metal)"),
    (axes[1,0], mixup,   "MixUp (λ = 0.5)"),
    (axes[1,1], cutmix,  "CutMix (parche de B en A)"),
]
for ax, img, title in panels:
    ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1B5E20")
    ax.axis("off")
plt.tight_layout(pad=1.0)
out = "presentacion_recursos/graficas_cuadernos_2_y_4/mixup_cutmix_visualizacion.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("Guardado:", out)
