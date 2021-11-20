# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torch
import torchvision.models as models

import base.ham10000_autoconfig
from exp8.ham10000_dataset_splitter import Ham10000DatasetSplitter
from exp8.ham10000_resnet18_validator import Ham10000ResNet18Validator
from exp8.ham10000_resnet18_predictor import Ham10000ResNet18Predictor


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    print('1 - Splits training, validation and test sets')
    metadata_path = base.ham10000_autoconfig.get_metadata_path()
    images_path = base.ham10000_autoconfig.get_images_path()

    splitter = Ham10000DatasetSplitter(metadata_path, images_path, percent_val=.15, percent_test=.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - Load ResNet18 model')
    model = models.resnet18()
    resnet18_parameters_path = base.ham10000_autoconfig.get_resnet18_parameters_path()
    #resnet18_parameters_filename = '20211110212929_ham10000_trained_model.pth'
    # resnet18_parameters_filename = '20211102005954_ham10000_trained_model.pth' # (weighted)
    # resnet18_parameters_filename = '20211030161151_ham10000_trained_model.pth' # (baseline - no weighted)
    # resnet18_parameters_filename = '20211108145111_ham10000_trained_model.pth'
    resnet18_parameters_filename = '20211110212929_ham10000_trained_model.pth' # albumentation-1

    resnet18_parameters = resnet18_parameters_path + resnet18_parameters_filename
    model.load_state_dict(torch.load(resnet18_parameters))

    log_time('\tValidation start time:')
    print('3 - validate model')
    validator = Ham10000ResNet18Validator(model, validation_dataloader)
    validator.run_validation()
    log_time('\tValidation end time:')

    print('5 - make predictions')
    predictor = Ham10000ResNet18Predictor(model, test_dataloader)
    predictor.run_predictor()

    log_time('Done!')
