import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pywt import dwt2
from sklearn import svm
import torch
import numpy as np
import os
import cv2
#os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
os.chdir(r'C:\Users\User\Desktop\paper_recreation')
from DCT_DWT_SVM_DEBUG import load_data
from PCA_CNN import PCA, sanity_svm
import scipy.io

def load(n_train, n_test):
    #############
    # input the number of desired train and test samples then this program spits out two tuples with PCA processed
    # results
    #
    # n_train: number of train examples (int)
    # n_train: number of test examples (int)
    # output: train, test
    #   train = [(data[0], target[0]), (data[1], target[1]), ... (data[n_train], target[n_train])]
    #   test = [(data[0], target[0]), (data[1], target[1]), ... (data[n_test], target[n_test])]
    ##############

    dirs = {'MNIST_train': './data/MNIST/processed/train.pt',
            'MNIST_test': './data/MNIST/processed/test_2.pt',
            'MNIST_train_label': './data/MNIST/processed/train_label.pt',
            'MNIST_test_label': './data/MNIST/processed/test_label.pt'}

    # determine necessary loading options such that PCA is trained on same data (otherwise it won't be separable)
    c_all = True
    if n_train+n_test < 60000:
        c_all = False
    elif n_train+n_test > 70000:
        print("there are only 70,000 samples to choose from, retry")
        return

    # load the normalized [0,1] data from their respective folders. the normalization follows Singh
    if c_all:
        data = torch.cat((torch.load(dirs['MNIST_train']), torch.load(dirs['MNIST_test'])), 0)
        targets = torch.cat((torch.load(dirs['MNIST_train_label']), torch.load(dirs['MNIST_test_label'])), 0)
    else:
        data = torch.load(dirs['MNIST_train'])
        targets = torch.load(dirs['MNIST_train_label'])

    # perform PCA
    data = PCA.my_pca(data)

    # make into a organized tuples and return
    train = [(np.asarray(data[i]), int(targets[i])) for i in range(n_train)]
    test = [(np.asarray(data[i]), int(targets[i])) for i in range(n_train, n_train+n_test)]
    return train, test

def run_me(n_train, n_test):
    train, test = load(n_train, n_test)
    train_data = [item[0] for item in train]
    train_l = [item[1] for item in train]
    test_data = [item[0] for item in test]
    test_l = [item[1] for item in test]
    return (train_data, test_data), (train_l, test_l)


data, targets = run_me(10000, 10000)
print('data got')
sanity_svm.test_svm(data, targets)