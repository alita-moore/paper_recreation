import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')
from PCA_CNN import PCA

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
        # self.EMNIST_train = self.__getitem__('EMNIST_train')
        # self.EMNIST_test = self.__getitem__('EMNIST_test')
        # self.EMNIST_train_label = self.__getitem__('EMNIST_train_label')
        # self.EMNIST_test_label = self.__getitem__('EMNIST_test_label')
        self.MNIST_train = self.__getitem__('MNIST')
        # self.MNIST_test = self.__getitem__('MNIST_test')


    def __getitem__(self, fn):
        dirs = {'MNIST_train': './data/MNIST/processed/train.pt',
            'MNIST_test': './data/MNIST/processed/test_2.pt',
            'MNIST_train_label': './data/MNIST/processed/train_label.pt',
            'MNIST_test_label': './data/MNIST/processed/test_label.pt'}

        # load the normalized [0,1] data from their respective folders. the normalization follows Singh
        data_train = torch.load(dirs[fn + '_train'])
        data_test = torch.load(dirs[fn + '_test'])
        data = PCA.my_pca(torch.stack([data_train, data_test], dim=0)) # NOT WORKING BECAUSE CANT CONCATENATE?
        target = torch.load(dirs[str(fn+'_train_label')]) + torch.load(dirs[str(fn+'_test_label')])

        train = [(data[i], target[i]) for i in range(self.n)]
        test = [(data[i], target[i]) for i in range(60000, 60000 + self.m)]
        result = (train, test)
        return result

    def __len__(self):
        pass

def uni_polar_sigmoid(output, target):

    pass

# class Net(nn.module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#     def forward(self):
#         pass


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_train,
                                           batch_size=64,
                                           shuffle=True)


n = 66
input_size = n
target_size = 1
num_epochs = 200
learning_rate = 0.001


# define model
model = nn.Linear(input_size, target_size)

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
        # Forward pass
        inputs = item[0].reshape(-1, n).type(torch.float32)
        targets = item[1].reshape(-1, 1).type(torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    perf[epoch] = loss.item()
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        # print(np.shape(np.asarray(targets)))
        # print(np.shape(np.asarray(inputs)))

# # Plot the graph
# predicted = model(inputs).detach().numpy()
plt.plot(perf)
