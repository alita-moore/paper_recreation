from DCT_DWT_SVM_DEBUG import load_data, test_svm
import os
import numpy as np


# recollect_data = False
#
# # checks for data, if recollect data is True re-process data again
# if not os.path.exists(r'.\data\Processed_DCT_DWT\data.npy'):
#     data = load_data.load_data()
#     np.save(r'.\data\Processed_DCT_DWT\data', data)
# elif recollect_data:
#     data = load_data.load_data()
#     np.save(r'.\data\Processed_DCT_DWT\data', data)
# else:
#     data = np.load(r'.\data\Processed_DCT_DWT\data.npy', allow_pickle=True)


# number of data samples to use
n = 60000
data = load_data.load_data(n)

features = 196
sample_MNIST = (data[0][0:n], data[1])
label_MNIST = (data[2][0:n], data[3])
test_svm.test_svm(sample_MNIST, label_MNIST, features)