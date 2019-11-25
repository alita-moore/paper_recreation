import os

import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision
import numpy as np
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
        image = np.asarray(torch.squeeze(image))
        # plot and save image
        plt.imshow(image.astype(float), cmap='gray_r')
        plt.show()
        plt.savefig(save_dir + '/' + str(int(label[i])) + '_' + str(i) + '.png')
        plt.close()
        plt.clf()


# download MNIST
data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
train_data = data.train_data
train_label = data.train_labels

data = torchvision.datasets.MNIST(root='./data', train=False, download=True)
test_data = data.train_data
test_label = data.train_labels

# normalize data to -1 to 1 for tanh output and convert to fp32 (single precision)
train_data = (train_data.type(torch.float32) / 254)
test_data = (test_data.type(torch.float32) / 254)

# add channels dimension for pytorch
train_data = torch.unsqueeze(train_data, 1)
test_data = torch.unsqueeze(test_data, 1)

# save output images to check if data is well matched
save_output_images(train_data[0:100], './check_train', train_label[0:100])
save_output_images(test_data[0:100], './check_test', test_label[0:100])

# save data
torch.save(train_data, './data/MNIST/processed/train.pt')
torch.save(test_data, './data/MNIST/processed/test_2.pt')
torch.save(train_label, './data/MNIST/processed/train_label.pt')
torch.save(test_label, './data/MNIST/processed/test_label.pt')
