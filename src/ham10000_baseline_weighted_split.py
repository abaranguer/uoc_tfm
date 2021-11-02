# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from ham10000_dataset_weighted_splitter import Ham10000DatasetWeightedSplitter
from ham10000_resnet18_predictor import Ham10000ResNet18Predictor
from ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from ham10000_resnet18_validator import Ham10000ResNet18Validator

import numpy as np
import torch

def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    metadata_path_lnx = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    metadata_path_win = 'C:/albert/UOC/dataset/HAM10000_metadata'
    metadata_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata'
    images_path_lnx = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'
    images_path_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225/'
    images_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/300x225/'

    log_time('Start time:')
    np.random.seed(0)
    torch.manual_seed(0)

    print('1 . Splits training, validation and test sets')
    splitter = Ham10000DatasetWeightedSplitter(metadata_path_win, images_path_win, percent_val=0.15, percent_test=0.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    model = models.resnet18()

    print('3 - train model')
    trainer = Ham10000ResNet18Trainer(train_dataloader, model, epochs=5)

    log_time('\tTraining start time:')
    tensorboard_logs_lnx = '/home/albert/UOC-TFM/tensorboard-logs'
    tensorboard_logs_win = 'C:/albert/UOC/tensorboard-logs'
    tensorboard_logs_clb = '/content/drive/MyDrive/UOC-TFM/tensorboard-logs'

    writer = SummaryWriter(log_dir=tensorboard_logs_win)
    trainer.run_training(writer)
    log_time('\tTraining end time:')

    print('4 - validate model')
    validator = Ham10000ResNet18Validator(model, validation_dataloader)
    validator.run_validation()

    print('5 - make predictions')
    predictor = Ham10000ResNet18Predictor(model, test_dataloader)
    predictor.run_predictor()

    writer.close()

    log_time('Done!')

