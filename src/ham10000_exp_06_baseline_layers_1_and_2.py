# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

from torch.utils.tensorboard import SummaryWriter

import base.ham10000_autoconfig
from exp6.ham10000_dataset_weighted_splitter import Ham10000DatasetWeightedSplitter
from exp6.ham10000_resnet18_predictor import Ham10000ResNet18Predictor
from exp6.ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from exp6.ham10000_resnet18_validator import Ham10000ResNet18Validator
from exp6.ham10000_model import Model


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    log_time('Start time:')

    print('1 . Splits training, validation and test sets')
    metadata_path = base.ham10000_autoconfig.get_metadata_path()
    images_path = base.ham10000_autoconfig.get_images_path()
    splitter = Ham10000DatasetWeightedSplitter(metadata_path, images_path, percent_val=0.15, percent_test=0.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    model_2_layers = Model(2)

    print('3 - train model')
    model_2_layers.train()
    trainer = Ham10000ResNet18Trainer(train_dataloader, model_2_layers, epochs=5)

    log_time('\tTraining start time:')
    tensorboard_logs_path = base.ham10000_autoconfig.get_tensorboard_logs_path()
    writer = SummaryWriter(log_dir=tensorboard_logs_path)
    trainer.run_training(writer)

    log_time('\tTraining end time:')

    print('4 - validate model')
    validator = Ham10000ResNet18Validator(model_2_layers, validation_dataloader)
    validator.run_validation()

    print('5 - make predictions')
    predictor = Ham10000ResNet18Predictor(model_2_layers, test_dataloader)
    predictor.run_predictor()

    writer.close()

    log_time('Done!')