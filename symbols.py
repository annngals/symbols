# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:52 2020

@author: Anna Galsanova
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle
from skimage.filters import threshold_otsu
 
def lakes(image):
    B = ~image
    BB = np.ones((B.shape[0] + 2, B.shape[1] + 2))
    BB[1:-1, 1:-1] = B
    return np.max(label(BB)) - 1
 
def has_vline(image):
    lines = np.sum(image, 0) // image.shape[0]
    return 1 in lines

def has_hline(image):
    lines = np.sum(image, 1) // image.shape[1]
    return 1 in lines
 
def has_bay(image):
    b = ~image
    bb = np.zeros((b.shape[0] + 1, b.shape[1])).astype("uint8")
    bb[:-1, :] = b
    return lakes(~bb) - 1
 
def count_bays(image):
    holes = ~image.copy()
    return np.max(label(holes))
 
def recognize(region):
    lc = lakes(region.image)
    bays = count_bays(region.image)
    circ = region.perimeter ** 2 / region.area
    
    if lc == 2:
        if has_vline(region.image) and bays < 5:
            return "B"
        return "8"
    
    if lc == 1:
        if has_bay(region.image) > 0:
            return "A"
        if has_vline(region.image):
            if bays > 3:
                return "0"
            if (circ > 58):
                return "D"
            return "P"
        return "0"
    
    if lc == 0:
        if has_vline(region.image) and (bays == 0 or bays == 3):
            if np.all(region.image == 1):
                return "-"
            else:
                return "1"
        
        if bays == 2:
            return "/"
        
        if bays == 5:
            if has_hline(region.image):
                return '*'
            return 'W'
        else: 
            if bays == 4 and circ > 40:
                return "X"
            else:
                return "*"
    return None
 
image = plt.imread("symbols.png")
image = np.sum(image, 2)
image[image > 0] = 1
 
labeled = label(image)
# print(np.max(labeled))
 
regions = regionprops(labeled)
d = {}
for region in regions:
    symbol = recognize(region)
    if symbol not in d:
        d[symbol] = 1
    else:
        d[symbol] += 1
 
for key in d.keys():
    print(key)
    prc = d.get(key) / sum(d.values()) * 100
    print(prc, "%", "\n")
 
print(d)
print(sum(d.values()))
 
plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(labeled)
plt.show()