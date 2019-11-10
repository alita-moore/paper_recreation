import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pywt import dwt2
from sklearn import svm
import torch
import numpy as np
import os
import cv2
os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')



def zigzag(shape):
    # perform the same zig-zag assigment that's used in 71
    #
    # shape = tuple (x,x) for this paper (10,10) (can take a produce any square shape)
    # output = a list of index on a 10x10 matrix in zig zag order
    #
    # this index matrix can be used to quickly index a matrix equation and reassign values
    x = np.zeros((shape[0] * shape[1], 1))
    y = np.zeros((shape[0] * shape[1], 1))

    ###########
    # preamble
    ###########
    t = 0
    cond_x = True
    cond_y = True
    cond = True
    x_t = 0
    y_t = 0

    ############
    # zig zag until condition is met
    ############
    while cond:
        # condition verifications
        if x[t] == shape[0] - 1 and cond_x:
            x_t = t
            cond_x = False
        elif y[t] == shape[1] - 1 and cond_y:
            y_t = t
            cond_y = False
        elif not cond_x and not cond_y:
            cond = False

        # reassignments, t is ++ inside because of nested while loops
        if t == 0:
            y[t + 1] += 1
            t += 1
        elif x[t] == 0:
            if x[t - 1] == x[t] and y[t - 1] == y[t] - 1:
                while not y[t] == 0:
                    x[t + 1] = x[t] + 1
                    y[t + 1] = y[t] - 1
                    t += 1
            else:
                y[t + 1] = y[t] + 1
                t += 1
        elif y[t] == 0:
            if y[t - 1] == y[t] and x[t - 1] == x[t] - 1:
                while not x[t] == 0:
                    x[t + 1] = x[t] - 1
                    y[t + 1] = y[t] + 1
                    t += 1
            else:
                x[t + 1] = x[t] + 1
                t += 1

    ###########
    # Finish
    # Due to symmetry, the following mirror can be performed
    ###########

    # finish zig-zag values
    x_r = (shape[0] - 1) - x[0:y_t]
    y_r = (shape[1] - 1) - y[0:y_t]
    x[x_t + 1:shape[0] * shape[1]] = np.flip(x_r)
    y[x_t + 1:shape[0] * shape[1]] = np.flip(y_r)

    # fill a (shape) matrix with the corresponding index value
    result = np.zeros((10, 10))
    for i in range(100):
        result[int(x[i])][int(y[i])] = i

    return result

def load_data():
    #######################################
    # Preamble
    #######################################
    i_zig = zigzag((10, 10)).flatten()

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
            tmp = [tmp[i] for i in np.sort(np.argsort(tmp)[0:100])]

            # perform reassignment based on the zig-zag reassignment indices
            data[i][j[0]][0:10, 0:10] = np.asarray([tmp[int(i)] for i in i_zig]).reshape((10, 10))

    return data


# Saved notes
#############################
# when flattened the DWT and I believe DCT do not perform any compression for some reason (probably to do with
# their matrix formulation
#
# code for flattening...
# result = [[None]]*np.shape(Data)[0]
# for i in range(np.shape(Data)[0]):
#     result[i] = [two.flatten() for two in Data[i]]