# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

import base.ham10000_autoconfig
from exp7.ham10000_dataset_splitter import Ham10000DatasetSplitter
from exp7.ham10000_resnet18_trainer import Ham10000ResNet18Trainer


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    log_time('Start time:')
    np.random.seed(0)
    torch.manual_seed(0)
    metadata_path = base.ham10000_autoconfig.get_metadata_path()
    images_path = base.ham10000_autoconfig.get_images_path()
    print('1 . Splits training, validation and test sets')
    splitter = Ham10000DatasetSplitter(metadata_path,
                                       images_path,
                                       percent_val=0.02,
                                       percent_test=0.96,
                                       BATCH_SIZE=200,
                                       VAL_BATCH_SIZE=200,
                                       TEST_BATCH_SIZE=200)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader

    print('2 - create ResNet18 model')
    model = models.resnet18()
    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('3 - train model')
    trainer = Ham10000ResNet18Trainer(train_dataloader, epochs=500)
    log_time('\tTraining start time:')
    tensorboard_logs_path = base.ham10000_autoconfig.get_tensorboard_logs_path()
    writer = SummaryWriter(log_dir=tensorboard_logs_path)
    trainer.run_training_and_validation(model, loss, optimizer, writer, validation_dataloader)
    log_time('\tTraining end time:')

    writer.close()
    log_time('Done!')
