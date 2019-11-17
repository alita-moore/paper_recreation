import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pywt import dwt2
from sklearn import svm
import torch
import numpy as np
import os
import cv2
os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
from DCT_DWT_SVM_DEBUG import load_data
from PCA_CNN import PCA
import scipy.io

data = load_data.load_data(1000)

def load(fn):
    dirs = {'MNIST_train': './data/MNIST/processed/train.pt',
            'MNIST_test': './data/MNIST/processed/test_2.pt',
            'MNIST_train_label': './data/MNIST/processed/train_label.pt',
            'MNIST_test_label': './data/MNIST/processed/test_label.pt'}
    # load the normalized [0,1] data from their respective folders. the normalization follows Singh
    data = torch.load(dirs[fn])
    data = PCA.my_pca(data)
    targets = torch.load(dirs[str(fn + '_label')])
    result = [(np.asarray(data[i]), int(targets[i])) for i in range(1000)]
    return result

def run_me():
    train = load('MNIST_train')
    test = load('MNIST_test')
    train_data = [item[0] for item in train]
    train_l = [item[1] for item in train]
    test_data = [item[0] for item in test]
    test_l = [item[1] for item in test]
    return ((train_data, test_data), (train_l, test_l))
