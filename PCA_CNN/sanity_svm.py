import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics

os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
matplotlib.use('Agg')


def save_output_images(data, save_dir, label):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, image in enumerate(data):
        # # convert image to numpy for output
        # image = torch.squeeze(image).data.cpu().numpy()
        #
        # # move channel axis to match matplotlib standard of channels last
        # image = np.moveaxis(image, 0, -1)

        # plot and save image
        plt.imshow(image.astype(float), cmap='gray_r')
        plt.show()
        plt.savefig(save_dir + '/' + str(int(label[i])) + '_' + str(i) + '.png')
        plt.close()
        plt.clf()


def test_svm(sample, label, original):
    # sample: [train, test] images
    # label: [train, test] labels

    ###########################
    # SVM
    # There is no specified kernel used in the SVM, so I assume 'rbf' and gamma = 'scale'
    ###########################
    clf = svm.SVC(gamma='scale')

    # # after running, the train and test data matched as of 5:44pm 11/14
    # save_output_images(sample[0], './check_train_data', label[0])
    # save_output_images(sample[1], './check_test_data', label[1])

    # train
    t_start = time.time()
    train = [item for item in sample[0]]
    clf.fit(train, label[0])
    t_train = time.time() - t_start

    # test
    t_start = time.time()
    tmp = [clf.predict([two])[0] for two in sample[1]]
    t_test = time.time() - t_start

    # calculate accuracy
    expected = label[1]
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected, tmp)))

    # reporting on time to train and test
    print("Time to train: " + str(t_train) + \
          "\n\tAverage train time per item: " + str(t_train / float(len(label[1]))))
    print("Time to test: " + str(t_test) + \
          "\n\tAverage test time per item: " + str(t_test / float(len(label[1]))))
