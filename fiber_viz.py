import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def visualize_processing_steps(original, vesselness, binary, skeleton):
    """Affiche les 4 étapes clés du traitement."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original (MEB)")
    
    axes[1].imshow(vesselness, cmap='magma')
    axes[1].set_title("Filtre de Frangi")
    
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title("Seuillage Binaire")
    
    axes[3].imshow(skeleton, cmap='jet')
    axes[3].set_title("Squelette")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_annotations(image, annotations):
    """
    Affiche les fibres segmentées avec 5 points clés ordonnés.
    Règle : Vert (Start) -> ... -> Rouge (End)
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    colors = ['#00FF00', '#ADFF2F', '#FFFF00', '#FFA500', '#FF0000'] 
    
    for ann in annotations:
        kp = ann['keypoints']
        
        xs = [kp[i] for i in range(0, 15, 3)]
        ys = [kp[i+1] for i in range(0, 15, 3)]
        
        # 1. Tracer la polyline (ligne brisée) reliant les 5 points
        # On utilise une ligne plus épaisse et plus visible (cyan)
        plt.plot(xs, ys, '-', alpha=0.6, linewidth=2, c='cyan', zorder=4)
        
        # 2. Tracer les points individuels avec leur couleur respective
        for i in range(5):
            plt.scatter(xs[i], ys[i], c=colors[i], s=35, 
                        edgecolors='black', linewidths=0.5, zorder=10)

    plt.title(f"Visualisation 5-Kpts : {len(annotations)} fibres (Vert=Haut, Rouge=Bas)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def crop_bottom_info_bar(img, threshold=20):
    """
    Détecte et supprime le bandeau d'information noir en bas des images MEB.
    On suppose que le bandeau est une zone sombre continue sur toute la largeur.
    """
    height, width = img.shape[:2]
    
    for y in range(height - 1, 0, -1):
        row_mean = np.mean(img[y, :])
      
        if row_mean > threshold:
            crop_y = max(0, y - 5) 
            return img[0:crop_y, :], crop_y
            
    return img, height

def crop_percentage(img, percent=15):
    """
    Coupe un pourcentage fixe au bas de l'image.
    """
    height = img.shape[0]
    new_height = int(height * (1 - (percent / 100)))
    
    cropped_img = img[0:new_height, :]
    
    return cropped_img

# gray imagee reader
def read_gray(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_sam_masks(image, masks, alpha=0.6):
    """
    Superpose les masques d'instances générés par SAM sur l'image originale
    avec des couleurs aléatoires.
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        img_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_display = image.copy()
        
    height, width = image.shape[:2]
    
    mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    for mask_data in masks:
        mask = mask_data['segmentation'] # Masque binaire (True/False)
        
        # Choisir une couleur aléatoire RGB (0-255)
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        mask_overlay[mask] = color
        
    # Combiner l'image originale et le canevas de masques avec opacité (alpha)
    # img_combined = image * (1 - alpha) + mask_overlay * alpha
    img_combined = cv2.addWeighted(img_display, 1 - alpha, mask_overlay, alpha, 0)
    
    # Affichage avec Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_combined)
    plt.title(f"Visualisation SAM : {len(masks)} fibres segmentées")
    plt.axis('off') # Masquer les axes
    plt.show()
    
def visualize_fiber_cnn_style(image, annotations):
    """Affiche les 40 points avec un dégradé pour vérifier le sens de lecture."""
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    for ann in annotations:
        kp = ann['keypoints']
        xs = kp[0::3]
        ys = kp[1::3]
        
        # Trace la ligne de structure
        plt.plot(xs, ys, '-', color='cyan', linewidth=1, alpha=0.8)
        
        # Marqueurs de direction : Vert (Start) -> Rouge (End)
        plt.scatter(xs[0], ys[0], c='lime', s=15, zorder=5)  # Haut
        plt.scatter(xs[-1], ys[-1], c='red', s=15, zorder=5) # Bas

    plt.title(f"Visualisation FibeR-CNN (40 pts/fibre) - {len(annotations)} détectées")
    plt.axis('off')
    plt.show()
    
# --- À insérer dans ta boucle 'for mask_data in all_masks' ---

# Calcul de la courbure locale entre les points
def calculate_curvature(points):
    """Calcule la courbure moyenne d'une fibre à partir de ses points clés."""
    # Conversion en array pour calcul vectoriel
    pts = np.array(points) # Shape (40, 2)
    
    # Dérivées premières et secondes (différences finies)
    dx = np.gradient(pts[:, 1])
    dy = np.gradient(pts[:, 0])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Formule de la courbure : k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # On évite la division par zéro
    curvature_profile = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # On retourne la moyenne pour l'annotation globale
    return float(np.mean(curvature_profile))


def save_debug_diagnostic(img, all_masks, accepted_annotations, output_path):
    """
    Génère une image MEB montrant les fibres validées et celles rejetées.
    """
    # Création d'un calque de couleur
    debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = debug_img.copy()
    
    # On récupère les IDs des masques acceptés
    accepted_ids = [ann.get("metadata", {}).get("mask_index") for ann in accepted_annotations]

    for idx, mask_data in enumerate(all_masks):
        mask = mask_data['segmentation']
        
        if idx in accepted_ids:
            # VERT pour les fibres validées (40 pts extraits)
            color = [0, 255, 0] 
        else:
            # ROUGE pour les débris ou erreurs de squelette
            color = [0, 0, 255]
            
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

    # Fusion avec l'image originale pour la transparence
    cv2.addWeighted(overlay, 0.6, debug_img, 0.4, 0, debug_img)
    
    # Sauvegarde
    cv2.imwrite(output_path, debug_img)
    print(f"🔍 Diagnostic sauvegardé : {output_path}")
    

def mask_to_polygons(binary_mask):
    # Trouver les contours du masque
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Un polygone valide doit avoir au moins 3 points (x, y) -> 6 valeurs
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
    return polygons

from skimage.morphology import skeletonize  

def analyze_fiber_complexity(mask, image_gray):
    # 1. Détection des Beads (Amas)
    # On regarde si le masque est très large par rapport à son squelette
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_width = np.max(dist_transform)
    mean_width = np.mean(dist_transform[mask > 0])
    has_bead = bool(max_width > mean_width * 3) # Si un point est 3x plus large que la moyenne

    # 2. Détection du flou local
    x, y, w, h = cv2.boundingRect(mask)
    roi = image_gray[y:y+h, x:x+w]
    blur_score = cv2.Laplacian(roi, cv2.CV_64F).var()
    is_blurry = bool(blur_score < 100) # Seuil à ajuster selon tes images

    # 3. Détection des croisements (Junctions)
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    # On utilise un kernel de voisinage pour compter les voisins
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 10, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel)
    is_crossing = bool(np.any(neighbors > 12)) # Un point avec > 2 voisins

    return has_bead, is_blurry, is_crossing


def _as_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.copy()


def _save_or_show(fig, output_path=None, show=True):
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[VIZ] Figure sauvegardee: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_preprocessing_overview(original, variants, output_path=None, show=True):
    total = 1 + len(variants)
    fig, axes = plt.subplots(1, total, figsize=(5 * total, 5))
    if total == 1:
        axes = [axes]

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Originale")
    axes[0].axis("off")

    for ax, variant in zip(axes[1:], variants):
        ax.imshow(variant.image, cmap="gray")
        ax.set_title(f"Pretraitement\n{variant.name}")
        ax.axis("off")

    _save_or_show(fig, output_path=output_path, show=show)


def visualize_mask_summary(image, masks_by_variant, merged_masks, output_path=None, show=True):
    total = len(masks_by_variant) + 1
    fig, axes = plt.subplots(1, total, figsize=(5 * total, 5))
    if total == 1:
        axes = [axes]

    for ax, (variant_name, masks) in zip(axes, masks_by_variant.items()):
        overlay = _as_rgb(image)
        color_layer = np.zeros_like(overlay)
        rng = np.random.default_rng(42)
        for mask_data in masks:
            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            color_layer[mask_data["segmentation"]] = color
        blended = cv2.addWeighted(overlay, 0.55, color_layer, 0.45, 0)
        ax.imshow(blended)
        ax.set_title(f"{variant_name}\n{len(masks)} masks")
        ax.axis("off")

    overlay = _as_rgb(image)
    color_layer = np.zeros_like(overlay)
    rng = np.random.default_rng(7)
    for mask_data in merged_masks:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        color_layer[mask_data["segmentation"]] = color
    blended = cv2.addWeighted(overlay, 0.55, color_layer, 0.45, 0)
    axes[-1].imshow(blended)
    axes[-1].set_title(f"Fusion dedup\n{len(merged_masks)} masks")
    axes[-1].axis("off")

    _save_or_show(fig, output_path=output_path, show=show)


def visualize_single_fiber_debug(
    image,
    binary_mask,
    skeleton,
    keypoints,
    visibility,
    fiber_metrics,
    complexity_flags,
    title_suffix="",
    output_path=None,
    show=True,
):
    x, y, w, h = cv2.boundingRect(binary_mask.astype(np.uint8))
    margin = 12
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(image.shape[1], x + w + margin)
    y1 = min(image.shape[0], y + h + margin)

    roi = image[y0:y1, x0:x1]
    roi_mask = binary_mask[y0:y1, x0:x1]
    roi_skeleton = skeleton[y0:y1, x0:x1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(roi, cmap="gray")
    axes[0, 0].set_title("ROI brute")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_mask, cmap="gray")
    axes[0, 1].set_title("Masque")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(roi_skeleton, cmap="magma")
    axes[1, 0].set_title("Squelette")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(roi, cmap="gray")
    xs = [point[1] - x0 for point in keypoints]
    ys = [point[0] - y0 for point in keypoints]
    colors = ["lime" if v == 2 else "yellow" for v in visibility]
    axes[1, 1].plot(xs, ys, "-", color="cyan", linewidth=1.5, alpha=0.85)
    axes[1, 1].scatter(xs, ys, c=colors, s=18, edgecolors="black", linewidths=0.2)
    axes[1, 1].scatter(xs[0], ys[0], c="lime", s=40, edgecolors="black")
    axes[1, 1].scatter(xs[-1], ys[-1], c="red", s=40, edgecolors="black")
    axes[1, 1].set_title("40 keypoints")
    axes[1, 1].axis("off")

    metrics_text = (
        f"width={fiber_metrics['fiber_width']:.2f}px\n"
        f"length={fiber_metrics['fiber_length']:.2f}px\n"
        f"curvature={fiber_metrics['fiber_curvature']:.4f}\n"
        f"orientation={fiber_metrics['fiber_orientation']:.1f} deg\n"
        f"has_bead={complexity_flags['has_bead']}\n"
        f"is_blurry={complexity_flags['is_blurry']}\n"
        f"is_crossing={complexity_flags['is_crossing']}"
    )
    fig.suptitle(f"Debug fibre {title_suffix}".strip(), fontsize=12)
    fig.text(0.76, 0.12, metrics_text, fontsize=10, family="monospace")

    _save_or_show(fig, output_path=output_path, show=show)


def visualize_annotation_overview(image, annotations, output_path=None, show=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image, cmap="gray")

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=0.7, alpha=0.8)
        ax.add_patch(rect)

        kp = ann["keypoints"]
        xs = kp[0::3]
        ys = kp[1::3]
        vis = kp[2::3]
        ax.plot(xs, ys, "-", color="cyan", linewidth=0.8, alpha=0.7)
        point_colors = ["lime" if v == 2 else "orange" for v in vis]
        ax.scatter(xs, ys, c=point_colors, s=8, alpha=0.9)

    ax.set_title(f"Annotations finales - {len(annotations)} fibres")
    ax.axis("off")
    _save_or_show(fig, output_path=output_path, show=show)
