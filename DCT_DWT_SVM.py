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

# use SVM: scikitlearn
# clf = svm.SVC(gamma='scale')
# clf.fit(x,y)
# https://scikit-learn.org/stable/modules/svm.html


def load_data():
