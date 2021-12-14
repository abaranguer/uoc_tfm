# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torch
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

import base.ham10000_autoconfig
from exp8.ham10000_dataset_splitter import Ham10000DatasetSplitter
from exp8.ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from exp8.ham10000_resnet18_validator import Ham10000ResNet18Validator


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

    print('2 - train 5 epochs loading and saving epoch training')
    resnet18_parameters_path = base.ham10000_autoconfig.get_resnet18_parameters_path()
    timestamp = time.strftime("%Y%m%d")
    running_loss = 0.0
    for epoch in range(20):
        model = models.resnet18()
        loss = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

        trained_model_filename = resnet18_parameters_path + timestamp + f'_ham10000_trained_epoch_{epoch}.pth'
        if epoch > 0:
            previous_epoch = epoch - 1
            previous_trained_model_filename = resnet18_parameters_path + timestamp + f'_ham10000_trained_epoch_{previous_epoch}.pth'
            model.load_state_dict(torch.load(previous_trained_model_filename))

        print(f'3 - train 1 epoch (epoch {epoch})')
        trainer = Ham10000ResNet18Trainer(train_dataloader, epochs=1)
        log_time('\tTraining start time:')
        tensorboard_logs = base.ham10000_autoconfig.get_tensorboard_logs_path()
        writer = SummaryWriter(log_dir=tensorboard_logs)
        model.train()
        running_loss = trainer.run_training_by_epoch(
            model,
            loss,
            optimizer,
            writer,
            epoch,
            trained_model_filename,
            running_loss)
        log_time('\tTraining end time:')

        log_time('\tValidation start time:')
        print('3 - validate model')
        validator = Ham10000ResNet18Validator(validation_dataloader)
        model.eval()
        validator.run_epoch_validation(model, loss, writer, epoch)
        log_time('\tValidation end time:')

    writer.flush()
    log_time('Done!')
