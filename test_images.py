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

def compute_residuals(nat_dirlist):
    print('Computing residuals')
    imgs = []
    for img_path in tqdm.tqdm(nat_dirlist):
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]

    pool = Pool(cpu_count())   #Pool is used to parallelize the execution of the function extract_single
    w = pool.map(prnu.extract_single, imgs)  #w contains the noise residuals of the natural images
    pool.close()

    np.save("Residuals_512x512.npy", w)

    w = np.stack(w, 0)
    imgs = []
    return w

def plot_confusion_matrix(cm, name, fingerprint_devices):
    labels = []
    for elem in fingerprint_devices:
        labels.append(elem[:-2])
    labels_cm = [d.replace('Frontal', 'F').replace('Rear', 'R') for d in labels]

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 20})

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
    fig, ax = plt.subplots(figsize=(20,20))
    cax = disp.plot(cmap=plt.cm.Blues, xticks_rotation=90, ax=ax, values_format='.2f')
    plt.grid(False)

    disp.ax_.figure.axes[-1].yaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    disp.im_.set_clim(0, 1)

    # Increase the font size of x and y ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.tight_layout()
    plt.savefig('plots/'+name, format="pdf", pad_inches=5)
    plt.clf()
    plt.close()

# Horizontal Violin plot with increased font size and coloured violins
def plot_device(fingerprint_device, natural_indices, values, label):
    plt.style.use('default')
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=12)
    plt.figure(figsize=(8, 8))  # Adjust the values (width, height) as needed
    plt.title(str(fingerprint_device) + "'s fingerprint")
    plt.xlabel('Euclidean distance for query images')

    # Create a dictionary with the natural indices as keys and corresponding values as lists
    data = {}
    for idx, value in zip(natural_indices, values):
        if idx in data:
            data[idx].append(value)
        else:
            data[idx] = [value]

    # Convert the data dictionary to a list of lists
    data_list = [data[idx] for idx in np.unique(natural_indices)]

    # Create the violin plot
    parts = plt.violinplot(data_list, showmeans=True, showmedians=False, vert=False)

    # Set facecolor for violin plots
    for pc, lab in zip(parts['bodies'], np.unique(natural_indices)):
        other = fingerprint_device
        if "Frontal" in fingerprint_device:
            other = other.replace("Frontal", "Rear")
        elif "Rear" in fingerprint_device:
            other = other.replace("Rear", "Frontal")
        if (lab == fingerprint_device):
            pc.set_facecolor('red')
        elif (lab==other):
            pc.set_facecolor("orange")
        else:
            pc.set_facecolor('skyblue')

    # Set x-axis ticks and labels
    unique_indices = np.unique(natural_indices)
    if unique_indices is not None and len(unique_indices) > 0:
        ticks = range(1, len(unique_indices) + 1)
        labels = unique_indices
        plt.yticks(ticks, labels)

        # Set the tick label corresponding to the fingerprint_device to red text color
        for tick, lab in zip(ticks, labels):
            if lab == fingerprint_device:
                plt.gca().get_yticklabels()[tick - 1].set_color('red')

    plt.tight_layout()
    plt.savefig('plots/' + label + '/' + str(fingerprint_device) + '.pdf', format="pdf")
    plt.clf()
    plt.close()

def main():
    fingerprint_devices = os.listdir("test/Dataset/")
    fingerprint_devices = sorted(np.unique(fingerprint_devices))
    fingerprint_devices.remove('.DS_Store')

    nat_device = []
    for device in fingerprint_devices:
        nat_dirlist = np.array(sorted(glob('test/Dataset/' + device + '/Images/Natural/JPG/Test/*.jpg')))[:100]
        nat_device_sofar = np.array([os.path.split(i)[1].rsplit('_', 2)[0] for i in nat_dirlist])
        nat_device = np.concatenate((nat_device, nat_device_sofar))

    #Load fingerprints
    k = np.load("Fingerprints/Images/512x512.npy")
    k = np.stack(k, 0)

    nat_dirlist = []
    for device in fingerprint_devices:
        nat_dirlist = np.concatenate((nat_dirlist,np.array(sorted(glob('test/Dataset/' + device + '/Images/Natural/JPG/Test/*.jpg')))[:100]))

    #Compute residuals of test images
    w = compute_residuals(nat_dirlist)
    #w = np.load("Residuals_512x512.npy")
    #w = np.stack(w, 0)

    # Computing Ground Truth
    # gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
    # True means that the image is taken with the camera of the specific row
    gt = prnu.gt(fingerprint_devices, nat_device)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_devices), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        pce_values = []
        natural_indices = []
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = prnu.pce(cc2d)['pce']
            pce_values.append(prnu_pce)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
            natural_indices.append(nat_device[natural_idx][:-2])
        plot_device(fingerprint_devices[fingerprint_idx][:-2], natural_indices, pce_values, "PCE")

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    plt.style.use('seaborn')
    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
    print("Accuracy PCE ",f'{metrics.accuracy_score(gt.argmax(0), pce_rot.argmax(0)):.3f}')
    roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
    roc_curve_pce.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_pce.pdf')

    #Confusion matrix
    cm_pce = confusion_matrix(gt.argmax(0), pce_rot.argmax(0))
    plot_confusion_matrix(cm_pce, "Confusion_matrix_PCE.pdf", fingerprint_devices=fingerprint_devices)

if __name__ == '__main__':
    main()
