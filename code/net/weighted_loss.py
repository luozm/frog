import matplotlib.pyplot as plt
import copy
from skimage.morphology import convex_hull_image, skeletonize
from skimage import data, img_as_float, measure
from skimage.util import invert
from scipy import ndimage
from skimage.filters import gaussian
import numpy as np


def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1])
          | (pad[1:-1,1:-1] != pad[2:,1:-1])
          | (pad[1:-1,1:-1] != pad[1:-1,:-2])
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
          | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
          | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
          | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


cell = np.load('data/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.npy')
# The original image is inverted as the object must be white.
cell[cell != 58] = 0
cell[cell == 58] = 1

# class weight
c1 = 1/np.sum(cell == 1) + 0.2
c0 = 1/np.sum(cell == 0)

image = cell

contours = mask_to_outer_contour(image)
convex_hull = convex_hull_image(image)
concave = convex_hull & ~image
skeleton = skeletonize(image) | skeletonize(concave)
dist = ndimage.distance_transform_edt(~skeleton)
dist = 1 - dist/np.max(dist)
distance = copy.copy(dist)
H = contours | mask_to_outer_contour(convex_hull)
dist[H == 0] = 0
dist = gaussian(dist, sigma=4, mode='mirror')

# class weighted
dist[cell == 1] += c1
dist[cell == 0] += c0

dist = dist/np.max(dist)

fig, axes = plt.subplots(2, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title('Original picture')
ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('Loss picture')
ax[1].imshow(dist, cmap=plt.get_cmap('jet'), interpolation='nearest')
ax[1].set_axis_off()

ax[2].set_title('Skeleton picture')
ax[2].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_axis_off()

ax[3].set_title('Distance picture')
ax[3].imshow(distance, cmap=plt.get_cmap('jet'), interpolation='nearest')
ax[3].set_axis_off()

plt.tight_layout()
plt.show()
print()
