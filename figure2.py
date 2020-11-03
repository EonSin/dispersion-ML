# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:49:12 2019

@author: PITAHAYA
"""

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

img = Image.open('grey_input_for_ppt.png').rotate(90)
img = np.array(img)
img[60:,:4] = 255 # strange black box
img = Image.fromarray(img)

plt.figure()
plt.imshow(img, cmap='gray')
plt.yticks([1,8,16,24,32,40,48,56,64])
plt.xticks([1,8,16,24,32,40,48,56,64])
plt.xlabel('Pixel Bin Number, Logarithmic Period')
plt.ylabel('Pixel Bin Number, Detrended Group Velocity')
ax = plt.gca()
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')
plt.savefig('LGB_1034-5083_input.pdf', bbox_inches='tight', transparency=True)
plt.show()