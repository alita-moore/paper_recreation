import torch
import numpy as np
from scipy import ndimage
import cv2

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
index = [0, 1, 4, 5]  # list of image indice such that the labels are not filtered
for i in index:
    # make images binary using Otsu's method
    Data[i] = (Data[i] > -0.75).astype(np.uint8)
    length = len(Data[i])
    print(i)

    # clean images using opening morphology technique on a 2x2 kernel
    for j in enumerate(Data[i]):
        # process
        Data[i][j[0]] = cv2.morphologyEx(Data[i][j[0]].astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2)))
        if j[0] % round(length/10) == 0:
            print(j[0]/length*100)


























