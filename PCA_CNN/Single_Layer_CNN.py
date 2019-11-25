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

    def __len__(self):
        pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n = 66

        # Inputs to hidden layer of 99 nodes (as specified by the article)
        self.hidden = nn.Linear(self.n, 99)
        self.hidden.weight.data.fill_(random.random()-0.5)
        self.output = nn.Linear(99, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # with torch.no_grad():
        #     self.hidden.weight = torch.nn.Parameter(torch.from_numpy(np.random.rand(64, 99)-0.5).type(torch.float32))

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = x.softmax(0)
        return x


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_train,
                                           batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_test,
                                           batch_size=64,
                                           shuffle=True)

n = 66
input_size = n
target_size = 1
num_epochs = 100
learning_rate = 0.001


# define model
model = Net()

######
# loss and optimizer
# Singh uses a "uni-polar sigmoid activation function"
######
criterion = nn.Sigmoid()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

perf = [None]*num_epochs
# train the model

for epoch in range(num_epochs):
    for item in train_loader:
        if np.shape(item[0])[0] == 64:
            # Forward pass
            inputs = item[0].reshape(-1, n).type(torch.float32)
            targets = item[1].reshape(-1, 1).type(torch.float32)
            outputs = model.forward(inputs)
            loss = criterion(outputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    perf[epoch] = loss.mean().item()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: '.format(epoch + 1, num_epochs, loss.mean().item()))
        # print(np.shape(np.asarray(targets)))
        # print(np.shape(np.asarray(inputs)))

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for item in test_loader:
        inputs = item[0].reshape(-1, n).type(torch.float32)
        targets = item[1].reshape(-1, 1).type(torch.float32)
        output = model(inputs)
        test_loss += F.loss(output, targets, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred).sum().item)

    test_loss /= 10000

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 10000,
        100. * correct / 10000))


# # Plot the graph
# predicted = model(inputs).detach().numpy()
plt.plot(perf)
