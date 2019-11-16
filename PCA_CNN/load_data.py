import cv2
import numpy as np
import torch
from pywt import dwt2
from scipy.fftpack import dct

from DCT_DWT_SVM import zigzag


def load_data(n):
    #######################################
    # Preamble
    #######################################
    i_zig = zigzag.zigzag((14, 14))

    #######################################
    # Importing Downloaded raw files
    #######################################
    dirs = ['./data/MNIST/processed/train.pt',
            './data/MNIST/processed/test_2.pt',
            './data/MNIST/processed/train_label.pt',
            './data/MNIST/processed/test_label.pt']

    # load tensors
    data = [torch.load(item) for item in dirs]

    # transform into squeezed np arrays
    data = [np.squeeze(np.asarray(item)) for item in data]
    data[0] = data[0][0:n]
    data[1] = data[1][0:n]

    #######################################
    # Pre-Processing
    #######################################
    index = [0, 1] # list of image indice such that the labels are not filtered
    for i in index:
        # make images binary using Otsu's method (e.g. -0.5 is not arbitrary)
        print(i)
        data[i] = (data[i] > -0.5).astype(np.uint8).astype(np.float16)

        for j in enumerate(data[i]):

            # clean images using morphology technique on a 2x2 kernel (accomplishing the same goal as the 71 paper)
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
            data[i][j[0]][14:28, 14:28] = dwt2(data[i][j[0]], 'haar')[0]

            #######################################
            # DCT
            # They do no specify any specific kind of dct that's used
            # the pattern of
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
            #######################################

            # perform dct on image (ortho compresses, and dct is applied twice due to limitations of scipy function)
            # tmp = dct(dct(data[i][j[0]][14:28, 14:28].T, norm='ortho').T, norm='ortho').flatten()
            tmp = np.round(dct(dct(data[i][j[0]][14:28, 14:28].T, norm='ortho').T, norm='ortho')).flatten()
            data[i][j[0]][0:14, 14:28] = tmp.reshape(14, 14)

            # use the zig-zag calculations to reassign values to respective positions in flattened matrix
            tmp = np.asarray([tmp[int(i)] for i in i_zig[0, :]]).reshape(14, 14)

            # reassign back into the original matrix
            data[i][j[0]][0:14, 0:14] = tmp

    return data