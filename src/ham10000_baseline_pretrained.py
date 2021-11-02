# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torch
import torchvision.models as models

from ham10000_dataset_splitter import Ham10000DatasetSplitter
from ham10000_resnet18_validator import Ham10000ResNet18Validator


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    metadata_path_lnx = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    metadata_path_win = 'C:/albert/UOC/dataset/HAM10000_metadata'
    metadata_path_clb = '/cham10000_baseline.pyontent/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata'
    images_path_lnx = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'
    images_path_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225/'
    images_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/300x225/'
    resnet18_parameters_path_win = 'C:/albert/UOC/resnet18_parameters/'
    #resnet18_parameters_filename = '20211030161151_ham10000_trained_model.pth'
    resnet18_parameters_filename = '20211101003641_ham10000_trained_model.pth'
    print('1 - Splits training, validation and test sets')
    splitter = Ham10000DatasetSplitter(metadata_path_win, images_path_win, percent_val=.15, percent_test=.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - Load ResNet18 model')
    model = models.resnet18()
    resnet18_parameters = resnet18_parameters_path_win + resnet18_parameters_filename
    model.load_state_dict(torch.load(resnet18_parameters))

    log_time('\tValidation start time:')
    print('3 - validate model')
    validator = Ham10000ResNet18Validator(model, validation_dataloader)
    validator.run_validation()
    log_time('\tValidation end time:')

    log_time('Done!')
