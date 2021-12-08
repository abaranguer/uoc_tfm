# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import base.ham10000_autoconfig
from exp5.ham10000_dataset_weighted_splitter import Ham10000DatasetWeightedSplitter
from exp5.ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from exp5.ham10000_resnet18_validator import Ham10000ResNet18Validator


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    log_time('Start time:')
    np.random.seed(0)
    torch.manual_seed(0)

    print('1 . Splits training, validation and test sets')
    metadata_path = base.ham10000_autoconfig.get_metadata_path()
    images_path = base.ham10000_autoconfig.get_images_path()
    splitter = Ham10000DatasetWeightedSplitter(metadata_path, images_path, set_number=6, percent_val=0.15,
                                               percent_test=0.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    model = models.resnet18()

    print('3 - train model')
    trainer = Ham10000ResNet18Trainer(train_dataloader, epochs=9)

    log_time('\tTraining start time:')
    tensorboard_logs = base.ham10000_autoconfig.get_tensorboard_logs_path()
    writer = SummaryWriter(log_dir=tensorboard_logs)
    trainer.run_training(model, writer)
    log_time('\tTraining end time:')

    print('4 - validate model')
    validator = Ham10000ResNet18Validator(validation_dataloader)
    model.eval()
    validator.run_validation(model)

    print('5 - make predictions')
    validator = Ham10000ResNet18Validator(test_dataloader)
    model.eval()
    validator.run_validation(model)

    writer.close()

    log_time('Done!')
