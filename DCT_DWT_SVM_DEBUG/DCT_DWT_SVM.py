from scipy.fftpack import dct, idct
from pywt import dwt2
from sklearn import svm
import torch
import numpy as np
import cv2
import os
import time
# os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
os.chdir(r'C:\Users\User\Desktop\paper_recreation')


def zigzag(shape):
    ########################
    # (ONLY WORKS FOR SQUARE SHAPES, FURTHER CHANGES MAY BE USEFUL AND IS SUBJECT TO LATER DEVELOPMENT)
    # perform the same zig-zag assigment that's used in 71
    #
    # shape = tuple (x,x) for this paper (10,10) (can take and produce any square shape)
    # output = a list of appropriate indexes to assign from a flattened matrix
    #           (i.e. output = [0, 1, 10, 20 ...] for a 10x10 zigzag pattern. Given a 10x10 arbitrary matrix  A that's
    #           flattened with np.flatten(), and the zigzag matrix will be 1d B = [A[0], A[1], A[10], A[20] ...]
    #           The number of zigzag features chosen is directly controllable so B[0:100] will return 100 features
    #           in a zigzag order. Please note that the output is simply the appropriate index, and this function does
    #           not directly create B for reasons of optimization
    #########################

    ###########
    # preamble
    ###########
    x = np.zeros((shape[0] * shape[1], 1))
    y = np.zeros((shape[0] * shape[1], 1))
    t = 0
    cond_x = True
    cond_y = True
    cond = True
    x_t = 0
    y_t = 0

    ############
    # Build zig-zag pattern
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

    # Due to symmetry, the following mirror can be performed
    x_r = (shape[0] - 1) - x[0:y_t]
    y_r = (shape[1] - 1) - y[0:y_t]
    x[x_t + 1:shape[0] * shape[1]] = np.flip(x_r)
    y[x_t + 1:shape[0] * shape[1]] = np.flip(y_r)

    ###########
    # Create index matrix and return
    ###########

    # fill a linear matrix with the corresponding index in original flattened matrix corresponding to position
    result = np.zeros((1, int(shape[0]) * int(shape[1])))
    for i in range(shape[0] * shape[1]):
        result[0, i] = int(x[i])*shape[0] + int(y[i])

    return result

def load_data():
    #######################################
    # Preamble
    #######################################
    i_zig = zigzag((14, 14))

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
if not os.path.exists(r'.\data\Processed_DCT_DWT\data.npy'):
    os.makedirs(r'.\data\Processed_DCT_DWT')
    data = load_data()
    np.save(r'.\data\Processed_DCT_DWT\data', data)
else:
    data = np.load(r'.\data\Processed_DCT_DWT\data.npy', allow_pickle=True)

data = load_data()
print('data got')



def test(sample, label, features):
    # sample: [train, test] images
    # label: [train, test] labels

    ###########################
    # SVM
    # There is no specified kernel used in the SVM, so I assume 'rbf' and gamma = 'scale'
    ###########################
    clf = svm.SVC(gamma='scale')

    # train
    t_start = time.time()
    train = [item.flatten()[0:features] for item in sample[0]]
    clf.fit(train, label[0])
    t_train = time.time() - t_start

    # test
    t_start = time.time()
    tmp = [clf.predict([two]) for two in [one.flatten()[0:features] for one in sample[1]]]
    t_test = time.time() - t_start

    # calculate accuracy
    error = [1 if int(tmp[i]) == int(item) else 0 for i, item in enumerate(label[1])]
    accuracy = sum(error)/float(np.shape(label)[1])

    # reporting on time to train and test
    print("Time to train: " + str(t_train) + \
                   "\n\tAverage train time per item: " + str(t_train/float(np.shape(label)[1])))
    print("Time to test: " + str(t_test) + \
                   "\n\tAverage test time per item: " + str(t_test/float(np.shape(label)[1])))
    print("Overall Accuracy: " + str(accuracy))

n = 1000

# sample_EMNIST = [data[0][0:n], data[1][0:n]]
# label_EMNIST = [data[2][0:n], data[3][0:n]]
# test(sample_EMNIST, label_EMNIST)

features = 196
sample_MNIST = [data[4][0:n], data[5][0:n]]
label_MNIST = [data[6][0:n], data[7][0:n]]
test(sample_MNIST, label_MNIST, features)








