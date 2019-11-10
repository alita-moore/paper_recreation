import time

import numpy as np
from sklearn import svm


def test_svm(sample, label, features):
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
    tmp = [clf.predict([two]) for two in [one[0:14, 0:14].flatten()[0:features] for one in sample[1]]]
    t_test = time.time() - t_start

    # calculate accuracy
    error = [1 if int(tmp[i]) == int(item) else 0 for i, item in enumerate(label[1])]
    accuracy = sum(error) / float(np.shape(label)[1])

    # reporting on time to train and test
    print("Time to train: " + str(t_train) + \
          "\n\tAverage train time per item: " + str(t_train / float(np.shape(label)[1])))
    print("Time to test: " + str(t_test) + \
          "\n\tAverage test time per item: " + str(t_test / float(np.shape(label)[1])))
    print("Overall Accuracy: " + str(accuracy))
