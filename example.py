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


def main():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    ff_dirlist = np.array(sorted(glob('test/data/ff-jpg/*')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])

    nat_dirlist = np.array(sorted(glob('test/data/nat-jpg/*')))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(ff_device))
    k = []   #fingerprints of the cameras

    # for each device, we extract the images belonging to that device and we compute the corresponding PRNU, which is saved in the array k
    for device in fingerprint_device:
        imgs = []
        for img_path in ff_dirlist[ff_device == device]:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue
            im_cut = prnu.cut_ctr(im_arr, (720, 720, 3))
            imgs += [im_cut]
        k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]
    #Print fingerprints
    #i=0
    #for elem in k:
    #    cv.imshow(f'Prova_{i}', elem)
    #    cv.waitKey(0)
    #    i=i+1
    k = np.stack(k, 0)

    print('Computing residuals')

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (720, 720, 3))]

    pool = Pool(cpu_count())   #Pool is used to parallelize the execution of the function extract_single
    w = pool.map(prnu.extract_single, imgs)  #w contains the noise residuals of the natural images
    pool.close()

    w = np.stack(w, 0)

    # Computing Ground Truth
    # gt function return a matrix where the number of rows is equal to the number of cameras used for computing the fingerprints, and number of columns equal to the number of natural images
    # True means that the image is taken with the camera of the specific row
    gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']

    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        tn, tp, fp, fn = 0, 0, 0, 0  ###
        pce_values = []   ###
        natural_indices = []    ###
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = prnu.pce(cc2d)['pce']   ###
            pce_values.append(prnu_pce)   ###
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
            ###
            natural_indices.append(natural_idx)
            if fingerprint_device[fingerprint_idx] == nat_device[natural_idx]:
                if prnu_pce > 60.:
                    tp += 1.
                else:
                    fn += 1.
            else:
                if prnu_pce > 60.:
                    fp += 1.
                else:
                    tn += 1.
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        plt.title('PRNU for ' + str(fingerprint_device[fingerprint_idx]))
        plt.xlabel('query images')
        plt.ylabel('PCE')

        plt.bar(natural_indices, pce_values)
        plt.text(0.85, 0.85, 'TPR: ' + str(round(tpr, 2)) + '\nFPR: '+ str(round(fpr, 2)),
         fontsize=10, color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes)
        plt.axhline(y=60, color='r', linestyle='-')
        plt.xticks(natural_indices)
        plt.savefig('plots/' +str(fingerprint_device[fingerprint_idx])+'.png')

        plt.clf()
        ###

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
    print("Accuracy CC ",f'{metrics.accuracy_score(gt.argmax(0), cc_aligned_rot.argmax(0)):.3f}')
    print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))
    print("Accuracy PCE ",f'{metrics.accuracy_score(gt.argmax(0), pce_rot.argmax(0)):.3f}')
    roc_curve_cc = metrics.RocCurveDisplay(fpr=stats_cc['fpr'], tpr=stats_cc['tpr'], roc_auc=stats_cc['auc'], estimator_name='ROC curve')
    roc_curve_pce = metrics.RocCurveDisplay(fpr=stats_pce['fpr'], tpr=stats_pce['tpr'], roc_auc=stats_pce['auc'], estimator_name='ROC curve')
    plt.style.use('seaborn')
    roc_curve_cc.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_cc.png')
    roc_curve_pce.plot(linestyle='--', color='blue')
    plt.savefig('plots/' +'roc_curve_pce.png')

    '''residual = []
    residual.append(prnu.extract_single(prnu.cut_ctr(np.asarray(Image.open("test/data/nat-jpg/LG_D290_0_0083.jpg")), (512, 512, 3))))
    residual = np.stack(residual, 0)
    print('Computing cross correlation image')
    cc_aligned_rot = prnu.aligned_cc(k, residual)['cc']
    #print(cc_aligned_rot)
    index_max = max(range(len(cc_aligned_rot)), key=cc_aligned_rot.__getitem__)
    print("The image seems to be taken with ", fingerprint_device[index_max])'''

if __name__ == '__main__':
    main()
