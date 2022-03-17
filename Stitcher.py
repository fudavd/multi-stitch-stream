import copy
import datetime
from typing import List, Optional
import os
import sys
from src.utils.utils import Image, PanoramaImage
import cv2
import numpy as np
import src.Calibrate.StitchFunctions as StitchFunctions
from src.utils.Filters import mask_colors
from src.utils.Measures import obtain_color_range
from src.Calibrate.Stitcher import PairStitcher, MultiStitcher
from src.utils.utils import search_file_list

# im2 = Image("cam2")
# im1 = Image("cam1")
# stitcher = PairStitcher([im1, im2], mask_colors=["red", "blue", "black", "white"], save_res=True, verbose=True)
# stitcher.stitch()

image_list = search_file_list('./Calibration_data/Stitch/', '.png')
images = [Image(f'img{ind}', path) for ind, path in enumerate(image_list)]
verbose = True

multi_stitch = MultiStitcher(images, verbose=False)
multi_stitch.stitch()
multi_stitch.panorama.stitch_panorama("pareto")
global_h = multi_stitch.panorama.get_global_h()

images_p = multi_stitch.panorama.images
repetitions = 100
n_images = len(image_list)
indices = np.arange(n_images)

for counter in range(repetitions):
    ind = counter % n_images
    shuffle = (indices[1:n_images] + ind) % n_images
    # shuffle = np.random.choice(shuffle, len(shuffle), replace=False)
    images_shuffled = [images_p[ii] for ii in shuffle]
    global_h_shuffled = [global_h[ii] for ii in shuffle]
    pseudorama = PanoramaImage("pseudo", images_shuffled)
    pseudorama.set_global_h(global_h_shuffled)
    pseudorama.h, pseudorama.w = multi_stitch.panorama.image.shape[:2]
    pseudorama.stitch_panorama("linear_blend", "cross")
    result = pseudorama.panorama
    if verbose == True:
        cv2.imshow('test', cv2.resize(result, (720, 405)))
        cv2.waitKey(1)
    gap_img = images_p[ind]
    stitcher = PairStitcher([pseudorama, gap_img], save_res=True)
    stitcher.stitch(cov_min=0, cov_rat=0.1)

    # global_h[ind] = np.mean((stitcher.H, global_h[ind], global_h_iter[ind]), axis=0)
    global_h[ind] = np.mean((stitcher.H, global_h[ind]), axis=0)
    # global_h[ind] = stitcher.H

global_h_final = {}
global_h_test = []
image_list.sort()
images = [Image(f'img{ind}', path) for ind, path in enumerate(image_list)]
for image in images:
    for ind, image_p in enumerate(images_p):
        if np.all(image_p.image == image.image):
            global_h_final[image.name] = global_h[ind]
            global_h_test.append(global_h[ind])
            break


res = PanoramaImage("test", images)
res.set_global_h(global_h_test)
res.h, res.w = multi_stitch.panorama.image.shape[:2]
res.stitch_panorama("pareto")
result = res.panorama
cv2.imshow('test', cv2.resize(result, (720, 405)))
cv2.waitKey(0)

my_details = {
    'map_x': res.map_x,
    'map_y': res.map_y,
    'H_dict': global_h_final
}
output_dir = os.path.join("./Calibration_data", "Stitch")
np.savez(os.path.join(output_dir, "cal_data"), **my_details)

