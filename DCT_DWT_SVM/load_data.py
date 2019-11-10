import cv2
import numpy as np
import torch
from pywt import dwt2
from scipy.fftpack import dct

from DCT_DWT_SVM import zigzag


def load_data():
    #######################################
    # Preamble
    #######################################
    i_zig = zigzag.zigzag((14, 14))

    #######################################
    # Importing Downloaded raw files
    #######################################
    dirs = ['./data/EMNIST/sample/training_processed.pt',
            './data/EMNIST/sample/test_processed.pt',
            './data/EMNIST/label/train_label_processed.pt',
            './data/EMNIST/label/test_label_process.pt',
            './data/MNIST/processed/training_processed.pt',
            './data/MNIST/processed/test_processed.pt',
            './data/MNIST/processed/train_label_processed.pt',
            './data/MNIST/processed/test_label_process.pt']

    # load tensors
    data = [torch.load(item) for item in dirs]

    # transform into squeezed np arrays
    data = [np.squeeze(np.asarray(item)) for item in data]

    #######################################
    # Pre-Processing
    #######################################
    index = [0, 1, 4, 5]  # list of image indice such that the labels are not filtered
    for i in index:
        # make images binary using Otsu's method (e.g. -0.5 is not arbitrary)
        data[i] = (data[i] > -0.5).astype(np.uint8).astype(np.float16)

        # clean images using opening morphology technique on a 2x2 kernel (accomplishing the same goal as the 71 paper)
        for j in enumerate(data[i]):
            data[i][j[0]] = cv2.morphologyEx(data[i][j[0]].astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2)))

            #######################################
            # DWT
            #
            # Haar waves are used in 71
            # 'a single level 2D DWT of the input image is computer using the haar wavelet'
            # 'haar' is in the DWT package, type pywt.wavelist() for all wavelets
            # only the LL result was used in the paper (e.g. the [0] index, or cA in documentation linked below)
            #
            # https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html
            #######################################
            data[i][j[0]][0:14, 0:14] = dwt2(data[i][j[0]], 'haar')[0]

            #######################################
            # DCT
            # They do no specify any specific kind of dct that's used
            # the pattern of
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
            #######################################
            # perform dct and collect top 100 absolute value largest, then put in order of occurring (not biggest to least)
            tmp = dct(data[i][j[0]][0:14, 0:14]).flatten()
            tmp = np.asarray([tmp[int(i)] for i in i_zig[0, :]]).reshape(14, 14)

            # perform reassignment based on the zig-zag reassignment indices
            data[i][j[0]][0:10, 0:10] = tmp[0:10, 0:10]

    return data