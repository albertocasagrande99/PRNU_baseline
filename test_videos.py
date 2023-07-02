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

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, name, fingerprint_devices):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fingerprint_devices)
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, values_format='')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/'+name, pad_inches=5)
    plt.clf()

def plot_device(fingerprint_device, natural_indices, values, label, n_values):
    avgResult = []
    start_index = 0

    for size in n_values:
        end_index = start_index + size
        chunk = values[start_index:end_index]
        avg_result = np.average(chunk)
        avgResult.append(avg_result)
        start_index = end_index

    plt.title('PRNU for ' + str(fingerprint_device))
    plt.xlabel('query images')
    plt.ylabel(label)

    plt.bar(np.unique(natural_indices), avgResult)
    plt.xticks(np.unique(natural_indices), rotation=90)
    plt.tight_layout()
    plt.savefig('plots/'+ label + '/' +str(fingerprint_device)+'.png')

    plt.clf()


def main():
    fingerprint_devices = os.listdir("video_fingerprints/")
    fingerprint_devices = sorted(np.unique(fingerprint_devices))
    
    fingerprint_devices.remove('.DS_Store')

    nat_device = []
    for device in fingerprint_devices:
        nat_dirlist = np.array(sorted(glob('video_fingerprints/' + device + '/Nat/*.jpg')))
        nat_device_sofar = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_dirlist])
        nat_device = np.concatenate((nat_device, nat_device_sofar))

    k = np.load("PRNU_512x512.npy")
    k = np.stack(k, 0)

    n_values = []
    print('Computing residuals')
    nat_dirlist = []
    for device in fingerprint_devices:
        nat_dirlist = np.concatenate((nat_dirlist,np.array(sorted(glob('video_fingerprints/' + device + '/Nat/*.jpg')))))
        n_values.append(len(np.array(sorted(glob('video_fingerprints/' + device + '/Nat/*.jpg')))))

    print(nat_dirlist)
    imgs = []
    for img_path in tqdm.tqdm(nat_dirlist):
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

    pool = Pool(cpu_count())   #Pool is used to parallelize the execution of the function extract_single
    w = pool.map(prnu.extract_single, imgs)  #w contains the noise residuals of the natural images
    pool.close()

    w = np.stack(w, 0)
    imgs = []

    # Computing Ground Truth
    # gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
    # True means that the image is taken with the camera of the specific row
    gt = prnu.gt(fingerprint_devices, nat_device)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_devices), len(nat_device)))

    for fingerprint_idx, fingerprint_k in tqdm.tqdm(enumerate(k)):
        pce_values = []   ###
        natural_indices = []    ###
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = prnu.pce(cc2d)['pce']   ###
            pce_values.append(prnu_pce)   ###
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
            natural_indices.append(nat_device[natural_idx])
        plot_device(fingerprint_devices[fingerprint_idx], natural_indices, pce_values, "PCE", n_values=n_values)

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
    print("Accuracy PCE ",f'{metrics.accuracy_score(gt.argmax(0), pce_rot.argmax(0)):.3f}')
    roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
    plt.style.use('seaborn')
    roc_curve_pce.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_pce.png')

    #Confusion matrix
    cm_pce = confusion_matrix(gt.argmax(0), pce_rot.argmax(0))
    plot_confusion_matrix(cm_pce, "Confusion_matrix_PCE.png", fingerprint_devices=fingerprint_devices)

if __name__ == '__main__':
    main()
