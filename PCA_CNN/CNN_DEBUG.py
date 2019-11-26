import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
#os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
os.chdir(r'C:\Users\User\Desktop\paper_recreation')
from PCA_CNN import PCA
import random

####################
# note that (5) refers to
# "Digit Recognition Using Single Layer Neural Network with Principal Component Analysis", Singh, 2014
#
# According to fig. 4, the input layer has 66 features, the hidden layer has 99 nodes, and output layer has 10 nodes
# this is a fully connected neural network
####################


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.n = 60000
        self.m = 10000
        self.MNIST_train = self.__getitem__('MNIST', True)
        self.MNIST_test = self.__getitem__('MNIST', False)

    def __getitem__(self, fn, train):
        #############
        # input the number of desired train and test samples then this program spits out two tuples with PCA processed
        # results
        #
        # self.n: number of train examples (int)
        # self.n: number of test examples (int)
        # output: train, test
        #   train = [(data[0], target[0]), (data[1], target[1]), ... (data[self.n], target[self.n])]
        #   test = [(data[0], target[0]), (data[1], target[1]), ... (data[self.m], target[self.m])]
        ##############

        dirs = {'MNIST_train': './data/MNIST/processed/train.pt',
                'MNIST_test': './data/MNIST/processed/test_2.pt',
                'MNIST_train_label': './data/MNIST/processed/train_label.pt',
                'MNIST_test_label': './data/MNIST/processed/test_label.pt'}

        # determine necessary loading options such that PCA is trained on same data (otherwise it won't be separable)
        c_all = True
        if self.n + self.m < 60000:
            c_all = False
        elif self.n + self.m > 70000:
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
        if train:
            data = [(data[i], targets[i]) for i in range(self.n)]
        else:
            data = [(data[i], targets[i]) for i in range(self.n, self.n + self.m)]
        # result = (train, test)
        return data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(66, 99)
        self.output = nn.Linear(99, 10)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = x.sigmoid()
        return x

def validate(model, data, criterion):
    with torch.no_grad():
        for item in data:
            y = model(data[0])
            loss = criterion(y, data[1])
            return loss


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_train,
                                           batch_size=64,
                                           shuffle=True)  # outputs (sample[], targets[]) -> (64x66, 64x1)
test_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_test,
                                           batch_size=64,
                                           shuffle=True)  # outputs (sample[], targets[]) -> (64x66, 64x1)

n, target_size, num_epoc, learning_rate = 66, 1, 100, 0.001
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(num_epoc):
    for x in train_loader:
        # run model and collect loss
        y = model.forward(x[0].float())
        loss = criterion(y, x[1].float())

        # perform optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(str(i) + ': ' + str(loss))
