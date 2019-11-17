from sklearn.decomposition import PCA
import numpy as np
import re
import torch

# do PCA -> pca = PCA(n_components = 2)
# pca.fit(x) where x is a numpy array
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


def my_pca(inputs, *args):
    #####
    # data =  data [tensor]; as determined in ./PCA_CNN/Single_Layer_CNN.py
    #           label is ignored for this procedure, but maintained in the same location
    # return: data [tensor] where data has been processed
    #
    # The purpose of this function is to recreate the PCA steps as outlined in "Digit Recognition Using Single Layer
    # Neural Network with Principal Component Analysis" (Singh, 2014)
    #####

    #####
    # process inputs
    # Args of form 'n=###' will designate the number of features selected within the PCA decomposition
    #   note that defining n is OPTIONAL, otherwise n = 66 as is declared by Singh; this is included for debugging
    #   purposes
    # worth nothing that after n = 144 the extracted data becomes indistinguishable (low variance between samples)
    #####

    n = 66
    for i, item in enumerate(args):
        if 'n=' in item:
            n = int(re.findall(r'\d+', item)[0])


    ######
    # prodrome
    ######
    inputs = inputs.reshape(-1, 28*28)
    x = np.asarray(inputs.reshape(-1, 784))

    ##############
    # PCA implementation
    # paper follows specific PCA format, the following is that format
    #
    # 1) normalize variable around its mean value [expected value] (X)
    # 2) find the covariance of X (C) and get the eigenvector (V) and eigenvalue (D) of C
    # 3) select K most significant eigenvectors (K largest eigenvalues) and project X into K highest eigenvectors
    #       the paper uses 66 features, so K = 66
    #
    # note: I created this code by hand because I was struggling to confirm specific steps as outlined by the paper
    # in other more well known PCA protocols. I predict it follows the same behavior, but I wanted to be particular
    ###############

    # 1 #
    u = [np.mean(item) for item in x.transpose()]
    x = np.asarray([x[:, i]-item for i, item in enumerate(u)])

    # 2 #
    x = np.cov(x)
    x = np.linalg.eig(x)

    # 3 # numpy automatically sorts the most relevant eigenvectors
    eigens = np.real(x[1][0:n]).transpose()
    x = np.dot(np.asarray(inputs), eigens)  # inputs is m x f, eigens in f x n, output is m x n

    #########
    # wrap up
    #########

    return torch.from_numpy(x)
