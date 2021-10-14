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
import torchvision
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dx_decoder import dx_to_description
from dx_decoder import int_to_dx
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
    print('Loading net parameters...')
    model = models.resnet18()
    trained_model_filename = 'ham10000_trained_model.pth'
    model.load_state_dict(torch.load(trained_model_filename))
    model.eval()
    print('Parameters loaded')

    images = next(iter(test_dataloader))

    # print images
    output = torchvision.utils.make_grid(images['image'])

    print('Showing images:"')
    for dx, label, image_id in zip(images['dx'], images['label'], images['image_id']):
        print(f"\timage: {image_id}.jpg, dx: {dx}, {dx_to_description[dx]}: label: {label}")

    print('Tanca la finestra amb les quatre imatges de prova per a continuar!')
    imshow(output)

    # get the model predictions:
    with torch.no_grad():
        images_as_tensors = images['image']
        outputs = model(images_as_tensors)
        _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % int_to_dx[int(predicted[j])] for j in range(4)))


    # analyze full test set
    correct = 0
    total = 0

    for i, images in enumerate(test_dataloader, 0):
        inputs = images['image']
        labels = images['label']

        print(f'batch {i}')

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # calculate outputs by running images through the network
            outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 2500 test images: %d %%' % (100 * correct / total))