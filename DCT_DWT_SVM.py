from scipy.fftpack import dct
from pywt import dwt
from sklearn import svm
import torch
import numpy as np
import emnist

# use dct scipy.fftpack.dct
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

# use dwt: pywt.dwt()
# https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html

# use SVM: scikit-learn
# clf = svm.SVC(gamma='scale')
# clf.fit(x,y)
# https://scikit-learn.org/stable/modules/svm.html


def load_data():
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

    return Data


data = load_data()  # np array squeezed because other parts are not necessary
