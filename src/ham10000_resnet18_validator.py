# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch.optim
import torchvision
import torchvision.models as models
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from ham10000_dataset_loader import Ham10000Dataset
from ham10000_dx_decoder import dx_to_description
from ham10000_dx_decoder import int_to_dx


class Ham10000ResNet18Validator:
    def __init__(self, model, validation_dataloader):
        self.model = model
        self.validation_dataloader = validation_dataloader
        self.accuracy = 0.0

    def run_validation(self):
        correct = 0
        total = 0

        for i, images in enumerate(self.validation_dataloader, 0):
            inputs = images['image']
            labels = images['label']

            with torch.no_grad():
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                mAP = average_precision_score(labels.numpy(), predicted.numpy())

        self.accuracy = 100 * correct / total
        print(f'num of correct predicted images (True positives): {correct}')
        print(f'num of images : {total}')
        print(f'Accuracy of the network on the test images: {self.accuracy: .4f}%')






def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    print('Start')
    # 1. Load and normalize
    # lesion_id,image_id,dx,dx_type,age,sex,localization,dataset
    metadata_path_lnx = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    metadata_path_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000'
    metadata_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata'

    metadata_path = metadata_path_win
    df = pandas.read_csv(metadata_path)
    np.random.seed(0)  # set random seed, so I obtain a deterministic sequence of random numbers.
    train_set, aux_set = train_test_split(df, test_size=0.30)  # 70% 30%
    validation_set, test_set = train_test_split(aux_set, test_size=0.50)  # 15%, 15%

    images_path_lnx = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'
    images_path_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225'
    images_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/300x225/'

    image_folder = images_path_win

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Utilitzar la mitjana i la desviació típica dels canals RGB de les imatges de ham10000
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

    print('Showing images:')
    for dx, label, image_id in zip(images['dx'], images['label'], images['image_id']):
        print(f"\timage: {image_id}.jpg, dx: {dx}, {dx_to_description[dx]}: label: {label}")

    print('Tanca la finestra amb les imatges de prova per a continuar!')
    imshow(output)

    # get the model predictions:
    with torch.no_grad():
        images_as_tensors = images['image']
        outputs = model(images_as_tensors)
        _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % int_to_dx[int(predicted[j])] for j in range(4)))

    print('\n\nUsing Ham10000ResNet18Validator class')
    validator = Ham10000ResNet18Validator(model, test_dataloader)
    validator.run_validation()
