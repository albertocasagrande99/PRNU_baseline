# -*- coding: UTF-8 -*-
"""
@author: Alberto Casagrande (alberto.casagrande@studenti.unitn.it)
University of Trento 2023
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

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import random
from collections import Counter

def main():
    fingerprint_devices = os.listdir("test/Videos/")
    fingerprint_devices = sorted(np.unique(fingerprint_devices))
    fingerprint_devices.remove('.DS_Store')
    #for device in fingerprint_devices:
    #    if("Frontal" in device):
    #        fingerprint_devices.remove(device)
    # Create a mapping of device names to their corresponding index values
    device_to_index = {device: index for index, device in enumerate(fingerprint_devices)}

    k = np.load("Fingerprints/Videos/512x512_VideoLevel+.npy")
    k = np.stack(k, 0)

    cm = np.zeros((len(fingerprint_devices), len(fingerprint_devices)))

    for device in fingerprint_devices:
        videos = os.listdir("test/Videos/"+device+"/Videos/VideoLevel+/Test/")
        if('.DS_Store' in videos):
            videos.remove('.DS_Store')
        for video in videos:
            nat_device = []
            nat_dirlist = []
            imgs_list = np.array(sorted(glob('test/Videos/' + device + '/Videos/VideoLevel+/Test/' + video +'/*.jpg')))
            nat_list = imgs_list
            nat_device_sofar = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_list])
            nat_device = np.concatenate((nat_device, nat_device_sofar))
            nat_dirlist = np.concatenate((nat_dirlist,imgs_list))
            
            print('Computing residuals')
            imgs = []
            for img_path in tqdm.tqdm(nat_dirlist):
                imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]
            pool = Pool(cpu_count())   #Pool is used to parallelize the execution of the function extract_single
            w = pool.map(prnu.extract_single, imgs)  #w contains the noise residuals of the natural images
            pool.close()
            w = np.stack(w, 0)

            gt = prnu.gt(fingerprint_devices, nat_device)
            pce_rot = np.zeros((len(fingerprint_devices), len(nat_device)))
            for fingerprint_idx, fingerprint_k in tqdm.tqdm(enumerate(k)):
                for natural_idx, natural_w in enumerate(w):
                    cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
                    pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
            print('Computing statistics on PCE')
            stats_pce = prnu.stats(pce_rot, gt)
            
            # Count the occurrences of each class
            class_counts = Counter(pce_rot.argmax(0))

            # Find the class with the highest count (mode)
            major_class = class_counts.most_common(1)[0][0]

            print("Major class:", fingerprint_devices[major_class])
            true_class_index = device_to_index[device]
            cm[true_class_index,major_class]+=1

            print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
            print("Accuracy PCE ",f'{metrics.accuracy_score(gt.argmax(0), pce_rot.argmax(0)):.3f}')
            print(cm)

if __name__ == '__main__':
    main()
