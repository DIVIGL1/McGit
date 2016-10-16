# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 18:25:00 2016

@author: McSim
"""

import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from skimage.measure import compare_psnr
from time import time
#import pylab

def save_answer(ans, nr):
    with open("RES-W6-HW1-MP-ans"+str(nr)+".txt", "w") as fout:
        fout.write(str(ans))

min_colors_quantity = 8
max_colors_quantity = 20
        
image = imread('parrots.jpg')
# pylab.imshow(image) # показать фото

float_img = img_as_float(image)
del(image)

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(float_img.shape)
assert d == 3
image_array = np.reshape(float_img, (w * h, d))
image_array_mean = np.copy(image_array)
image_array_median = np.copy(image_array)

def PSNR(kmeans, n_colors):
    indexes = [0 for i in range(n_colors)]
    medians = [0 for i in range(n_colors)]
    means = [0 for i in range(n_colors)]
    for label in range(n_colors):
        indexes[label] = [i for i, elem in enumerate(image_array) if kmeans.labels_[i] == label]
        label_matrix = image_array[indexes[label]]
        medians[label] = np.median(label_matrix, axis=0)
        means[label] = np.mean(label_matrix, axis=0)
        image_array_mean[indexes[label]] = means[label]
        image_array_median[indexes[label]] = medians[label]

    mean_float_img = np.reshape(image_array_mean, original_shape)
    median_float_img = np.reshape(image_array_median, original_shape)
    
    a = compare_psnr(float_img, mean_float_img)
    b = compare_psnr(float_img, median_float_img)
    #print("For number of colors = {2} PSNR for mean = {0} and for median = {1}".format(a, b, n_colors))
    
    return([a,b])

comparing_resultats = []

for n_colors in range(min_colors_quantity, max_colors_quantity):
    print("Fitting model for number of colors = ", n_colors)
    t0 = time()
    kmeans = KMeans(n_clusters=n_colors, init='k-means++', random_state=241).fit(image_array)
    print("done in %0.3fs." % (time() - t0))
    del(t0)
    PSNR_ans = PSNR(kmeans, n_colors)
    #comparing_resultats.append(PSNR_ans)
    if PSNR_ans[0]>20 or PSNR_ans[1]>20:
        print("Minimum number of colors for PSNR > 20 is ", n_colors)
        break

save_answer(n_colors, 1)