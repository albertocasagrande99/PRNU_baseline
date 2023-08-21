# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018

Modified by:
Alberto Casagrande
Alessio Belli
University of Trento
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image

import prnu

import cv2 as cv
import time

import sys
import matplotlib.pyplot as plt
from sklearn import metrics
import tqdm
import random


def main():
    """
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    fingerprint_devices = os.listdir("test/Videos/")
    fingerprint_devices = sorted(np.unique(fingerprint_devices))
    fingerprint_devices.remove('.DS_Store')
    
    k = []   #fingerprints of the cameras
    for device in fingerprint_devices:
        print("Computing fingerprint of " + device)
        if (device != ".DS_Store"):
            imgs = []
            #ff_dirlist = np.array(sorted(glob('test/Dataset/' + device + '/Images/Flat/JPG/*.jpg')))     #for images
            ff_dirlist = np.array(sorted(glob('test/Videos/' + device + '/Videos/VideoLevel+/Train/*.jpg')))      #for videos
            #random.shuffle(ff_dirlist)
            #ff_dirlist = ff_dirlist[:50]
            for img_path in tqdm.tqdm(ff_dirlist):
                im = Image.open(img_path)
                im_arr = np.asarray(im)
                if im_arr.dtype != np.uint8:
                    print('Error while reading image: {}'.format(img_path))
                    continue
                if im_arr.ndim != 3:
                    print('Image is not RGB: {}'.format(img_path))
                    continue
                im_cut = prnu.cut_ctr(im_arr, (512, 512, 3))
                imgs += [im_cut]
            k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]
    np.save("512x512_VideoLevel+.npy", k)

if __name__ == '__main__':
    main()
