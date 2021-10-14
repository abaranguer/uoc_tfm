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
import torch.optim
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from ham10000_dataset_loader import Ham10000Dataset


def imshow(inp, title=None):
    # imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    print('Start')
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

    BATCH_SIZE = 40
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

    # 2. Define a ResNet Neural Network
    model = models.resnet18()

    # 3. Define a Loss function and optimizer
    loss = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    # select device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        # images = next(iter(test_dataloader))   <<<<   ATENCIÓ A AQUWEST CANVI, VALIDAR AIXÒ
        for i, images in enumerate(train_dataloader, 0):
            inputs = images['image']
            labels = images['label']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss_current = loss(outputs, labels)
            loss_current.backward()
            optimizer.step()

            # print statistics
            running_loss += loss_current.item()
            print(f'epoch: {epoch}; i : {i}')
            if i % 5 == 4:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # save trained model
    trained_model_filename = 'ham10000_trained_model.pth'
    torch.save(model.state_dict(), trained_model_filename)
    print('Done!')
