# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch.optim
import torchvision
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

import src.exp5.ham10000_autoconfig
from src.exp5.ham10000_dataset_loader import Ham10000Dataset
from src.exp5.ham10000_dx_decoder import int_to_dx


class Ham10000ResNet18Predictor:
    def __init__(self, model, test_dataloader):
        self.model = model
        self.test_dataloader = test_dataloader

    def run_predictor(self):
        images = next(iter(self.test_dataloader))

        with torch.no_grad():
            images_as_tensors = images['image']
            groun_truth_labels = images['label']
            outputs = self.model(images_as_tensors)
            _, predicted = torch.max(outputs, 1)

        title_pred = '   Predicted: ' + ' '.join('%5s' % int_to_dx[int(predicted[j])] for j in range(len(predicted)))
        # ground_truth = 'Ground Truth: ' + ' '.join('%5s' % int_to_dx[int(predicted[j])] for j in range(len(predicted)))

    def display_batch(self, images_batch, writer):
        grid_img = torchvision.utils.make_grid(images_batch, nrow=1, normalize=True, scale_each=True)
        self.matplotlib_imshow(grid_img)
        writer.add_image('Current image batch ', grid_img)

    def matplotlib_imshow(self, grid_img):
        np_grid_img = grid_img.numpy()
        plt.imshow(np.transpose(np_grid_img, (1, 2, 0)))


if __name__ == '__main__':
    print('Start')

    # lesion_id,image_id,dx,dx_type,age,sex,localization,dataset
    metadata_path = src.exp4.ham10000_autoconfig.get_metadata_path()
    df = pandas.read_csv(metadata_path)
    np.random.seed(0)  # set random seed, so I obtain a deterministic sequence of random numbers.
    train_set, aux_set = train_test_split(df, test_size=0.30)  # 70% 30%
    validation_set, test_set = train_test_split(aux_set, test_size=0.50)  # 15%, 15%

    image_folder = src.exp4.ham10000_autoconfig.get_images_path()

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
            [0.141, 0.152, 0.169])  # std. dev. of RGB channels of HAM10000 dataset
    ])

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

    print('\n\nUsing Ham10000ResNet18Predictor class')
    predictor = Ham10000ResNet18Predictor(model, test_dataloader)
    predictor.run_predictor()
