# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torchvision.models as models

from ham10000_dataset_splitter import Ham10000DatasetSplitter
from ham10000_resnet18_predictor import Ham10000ResNet18Predictor
from ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from ham10000_resnet18_validator import Ham10000ResNet18Validator


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    matadata_path = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    images_path = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'
    log_time('Start time:')

    print('1 . Splits training, validation and test sets')
    splitter = Ham10000DatasetSplitter(matadata_path, images_path)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    model = models.resnet18()

    print('3 - train model')
    trainer = Ham10000ResNet18Trainer(train_dataloader, model)

    log_time('\tTraining start time:')

    trainer.run_training()

    log_time('\tTraining end time:')

    print('4 - validate model')
    validator = Ham10000ResNet18Validator(model, validation_dataloader)
    validator.run_validation()

    print('5 - make predictions')
    predictor = Ham10000ResNet18Predictor(model, test_dataloader)
    predictor.run_predictor()

    log_time('Done!')
