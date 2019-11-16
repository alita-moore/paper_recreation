import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
os.chdir(r'C:\Users\Adria\PycharmProjects\paper_recreation\venv\paper_recreation')

####################
# note that (5) refers to
# "Digit Recognition Using Single Layer Neural Network with Principal Component Analysis", Singh, 2014
#
# According to fig. 4, the input layer has 66 features, the hidden layer has 99 nodes, and output layer has 10 nodes
# this is a fully connected neural network
####################


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # self.EMNIST_train = self.__getitem__('EMNIST_train')
        # self.EMNIST_test = self.__getitem__('EMNIST_test')
        # self.EMNIST_train_label = self.__getitem__('EMNIST_train_label')
        # self.EMNIST_test_label = self.__getitem__('EMNIST_test_label')
        self.MNIST_train = self.__getitem__('MNIST_train')
        # self.MNIST_test = self.__getitem__('MNIST_test')
        # self.MNIST_test_label = self.__getitem__('MNIST_test_label')

    def __getitem__(self, item):
        dirs = {'EMNIST_train' : r'./data/EMNIST/sample/training_processed_(0-1).pt',
                'EMNIST_test' : r'./data/EMNIST/sample/test_processed_(0-1).pt',
                'EMNIST_train_label' : r'./data/EMNIST/label/train_label_processed_(0-1).pt',
                'EMNIST_test_label' : r'./data/EMNIST/label/test_label_process_(0-1).pt',
                'MNIST_train' : r'./data/MNIST/processed/training_processed_(0-1).pt',
                'MNIST_test' : r'./data/MNIST/processed/test_processed_(0-1).pt',
                'MNIST_train_label' : r'./data/MNIST/processed/train_label_processed_(0-1).pt',
                'MNIST_test_label' : r'./data/MNIST/processed/test_label_process_(0-1).pt'}

        return torch.load(dirs[item])

    def __len__(self):
        pass


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset.MNIST_train,
                                           batch_size=64,
                                           shuffle=True)

# inputs = custom_dataset.MNIST_train
# targets = custom_dataset.MNIST_train_label
#
# hyper-parameters
input_size = 28*28
output_size = 1
num_epochs = 10
learning_rate = 0.001

# define model
model = nn.Linear(input_size, output_size)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    for item in train_loader:
        # Forward pass
        inputs = item[0].reshape(-1, 28*28)
        targets = item[1].reshape(-1, 1).type(torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 2 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# # Plot the graph
# predicted = model(inputs).detach().numpy()
