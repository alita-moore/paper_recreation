import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import os
import torchvision.transforms.functional as TF
import emnist
matplotlib.use('Agg')


def save_output_images(data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, image in enumerate(data):
        # convert image to numpy for output
        image = torch.squeeze(image).data.cpu().numpy()

        # move channel axis to match matplotlib standard of channels last
        image = np.moveaxis(image, 0, -1)

        # plot and save image
        plt.imshow(image, cmap='gray_r')
        plt.show()
        plt.savefig(save_dir + '/' + str(i) + '.png')
        plt.clf()


# download datasets
torchvision.datasets.MNIST(root='./data', train=True, download=True)
torchvision.datasets.MNIST(root='./data', download=True)

# use the emnist package to access, it seems like the paper uses 'letters' defend in report
# import emnist
# images, labels = emnist.extract_training_samples('letters')
# images, labels = emnist.extract_test_samples('letters')

train_data, train_label =  torch.load('./data/MNIST/processed/training.pt') # [torch.from_numpy(item) for item in emnist.extract_training_samples('letters')]  # torch.load('./data/MNIST/processed/training.pt') #
test_data, test_label = torch.load('./data/MNIST/processed/test.pt')  # [torch.from_numpy(item) for item in emnist.extract_test_samples('letters')]  #

# normalize data to -1 to 1 for tanh output and convert to fp32 (single precision)
train_data = (train_data.type(torch.float32) / 127.5) - 1
test_data = (test_data.type(torch.float32) / 127.5) - 1

# rotate data 90 degrees
train_data = train_data.rot90(k=1, dims=(1, 2))
train_data = train_data.flip(dims=(0, 1))

test_data = test_data.rot90(k=1, dims=(1, 2))
test_data = test_data.flip(dims=(0, 1))

# add channels dimension for pytorch
train_data = torch.unsqueeze(train_data,1)
test_data = torch.unsqueeze(test_data, 1)

# confirm shape of data
print(np.shape(train_data))
print(np.shape(test_data))

# check max and min for output activation and dtype
print(train_data.max())
print(train_data.min())
print(train_data.type())

save_output_images(train_data[:100, :, :], './check_train_data')
save_output_images(test_data[:100, :, :], './check_test_data')
torch.save(train_data, './data/MNIST/processed/training_processed.pt')
torch.save(test_data, './data/MNIST/processed/test_processed.pt')
torch.save(train_label, './data/MNIST/processed/train_label_processed.pt')
torch.save(test_label, './data/MNIST/processed/test_label_process.pt')


