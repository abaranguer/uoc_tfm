# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

'''
https://pytorch.org/hub/pytorch_vision_resnet/
https://pytorch.org/vision/stable/models.html
https://www.programcreek.com/python/example/108007/torchvision.models.resnet18

Codi base de
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torchvision
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim


from ham10000_dataset_loader import Ham10000Dataset


def imshow(inp, title=None):
    # imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    # 1. Load and normalize
    # lesion_id,image_id,dx,dx_type,age,sex,localization,dataset
    df = pandas.read_csv("/home/albert/UOC-TFM/dataset/HAM10000_metadata")
    train_set, test_set = train_test_split(df, test_size=0.25)

    image_folder = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    '''
    '# training data
    train_data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    '''

    train_dataset = Ham10000Dataset(train_set, image_folder, train_data_transform)
    test_dataset = Ham10000Dataset(test_set, image_folder, train_data_transform)

    BATCH_SIZE = 10
    TEST_BATCH_SIZE = 4

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True
    )

    images = next(iter(test_dataloader))

    output = torchvision.utils.make_grid(images['image'])

    print('Showing images:"')
    for test_image_name in images['label']:
        print(f"\timage: {test_image_name}")

    imshow(output)

    # 2. Define a Convolutional Neural Network
    model = models.resnet18()

    # 3. Define a Loss function and optimizer
    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    # select device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # save trained model
    trained_model_filename = 'ham10000_trained_model.pth'
    torch.save(model.state_dict(), trained_model_filename)

    # TODO
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # 5. Test the network on the test data
    #
    # MORE THINGS TO DO
    # Training on GPU
    #
    # Just like how you transfer a Tensor onto the GPU, you transfer the neural net onto the GPU.
    # Let’s first define our device as the first visible cuda device if we have CUDA available:
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # Assuming that we are on a CUDA machine, this should print a CUDA device:
    #
    # print(device)
    #
    # Out:
    #
    # cuda:0
    #
    # The rest of this section assumes that device is a CUDA device.
    #
    # Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:
    #
    # net.to(device)
    #
    # Remember that you will have to send the inputs and targets at every step to the GPU too:
    #
    # inputs, labels = data[0].to(device), data[1].to(device)
    #     #
    # Training on multiple GPUs
    #
    # If you want to see even more MASSIVE speedup using all of your GPUs, please check out Optional: Data Parallelism.

