import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import filters, color
from skimage.morphology import binary_opening, binary_dilation, disk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.color import rgb2lab
from contour import process_flower_image

NAMED_COLORS = {
    "Rouge":   [53, 80, 67],
    "Orange":  [63, 43, 74],
    "Jaune":   [97, -21, 94],
    "Vert":    [36, -43, 41],
    "Bleu":    [32, 79, -107],
    "Violet":  [29, 58, -36],
    "Rose":    [66, 64, 17],
    "Blanc":   [100, 0, 0],
    "Magenta": [60, 93, -60],
    "Cyan":    [91, -48, -14],
}

def name_color(lab):
    return min(NAMED_COLORS, key=lambda c: np.linalg.norm(lab - NAMED_COLORS[c]))

def score(r):
    dist = (r.centroid[0] - cy)**2 + (r.centroid[1] - cx)**2
    return dist / r.area

script_dir = os.path.dirname(os.path.abspath(__file__))
fleur_rgb = imread(os.path.join(script_dir, 'test4.jpeg'))

#fleur_rgb = process_flower_image(fleur_rgb)

fleur_hsv = color.rgb2hsv(fleur_rgb)
fleur_S = fleur_hsv[:, :, 1]
fleur_V = fleur_hsv[:, :, 2]

thresh_s = filters.threshold_otsu(fleur_S)
thresh_v = filters.threshold_otsu(fleur_V)
fg_mask = (fleur_S >= thresh_s) | (fleur_V >= thresh_v)

fg_mask = binary_opening(fg_mask, disk(25))  # bigger — kills fragmented background noise
fg_mask = binary_fill_holes(fg_mask)
fg_mask = binary_dilation(fg_mask, disk(8))

labeled = label(fg_mask)
regions = regionprops(labeled)

cy, cx = fg_mask.shape[0] / 2, fg_mask.shape[1] / 2

best = min(regions, key=score)
fg_mask = labeled == best.label

fleur_no_bg = fleur_rgb.copy()
fleur_no_bg[~fg_mask] = 0

mean_lab = rgb2lab(fleur_no_bg).reshape(-1, 3)[fg_mask.ravel()]
print(name_color(mean_lab.mean(axis=0)))

plt.imshow(fleur_no_bg)
plt.show()