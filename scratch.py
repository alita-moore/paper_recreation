import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pywt import dwt
from sklearn import svm
import torch
import numpy as np
import emnist
import os
os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')


def load_data():
    def clean(A):
        shape = np.shape(A)
        pad = np.zeros((shape[0] + 2, shape[1] + 2))
        pad[1:shape[0] + 1, 1:shape[1] + 1] = A
        for i in range(1, shape[0] + 1):
            for j in range(1, shape[1] + 1):
                prox = [pad[i-1][j+1], pad[i][j+1], pad[i+1][j+1],
                       pad[i-1][j], pad[i][j], pad[i+1][j],
                       pad[i-1][j-1], pad[i][j-1], pad[i+1][j-1]]
                # surrounding area [top-left, top, top-right,
                #                   left, center, right,
                #                   below-left, below, below-right]

                if sum(prox) == 1:
                    A[i - 1][j - 1] = 0




    load_data.__doc__ = ("A function that loads pre-processed data into a numpy array \n"
                         " \n"
                         "Outputs 8x(dimension of data) tuple, which follows:\n"
                         "[EMNIST train, EMNIST test, EMNIST train label, EMNIST test label...\n"
                         ", MNIST train, MNIST test, MNIST train labels, MNIST test labels")
    dirs = ['./data/EMNIST/sample/training_processed.pt',
                 './data/EMNIST/sample/test_processed.pt',
                 './data/EMNIST/label/train_label_processed.pt',
                 './data/EMNIST/label/test_label_process.pt',
                 './data/MNIST/processed/training_processed.pt',
                 './data/MNIST/processed/test_processed.pt',
                 './data/MNIST/processed/train_label_processed.pt',
                 './data/MNIST/processed/test_label_process.pt']

    # load tensors
    Data = [torch.load(item) for item in dirs]

    # transform into squeezed np arrays
    Data = [np.squeeze(np.asarray(item)) for item in Data]

    # Pre-Processing
    #######################################
    # Binarization is performed on each sample in 71 using Otsu's method I find that anything about -0.75 should be 1
    index = [0, 1, 4, 5]  # list of image indice such that the labels are not filtered
    for i in index:
        Data[i] = (Data[i] > -0.75).astype(int)






    #############################
    # when flattened the DWT and I believe DCT do not perform any compression for some reason (probably to do with
    # their matrix formulation
    #
    # code for flattening...
    # result = [[None]]*np.shape(Data)[0]
    # for i in range(np.shape(Data)[0]):
    #     result[i] = [two.flatten() for two in Data[i]]

    return Data

data = load_data()
plot.hist()