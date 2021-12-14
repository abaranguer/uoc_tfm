# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

import base.ham10000_autoconfig
from exp6.ham10000_dataset_weighted_splitter import Ham10000DatasetWeightedSplitter
from exp6.ham10000_resnet18_model_with_dropout import Ham10000Resnet18ModelWithDropout2d
from exp6.ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from exp6.ham10000_resnet18_validator import Ham10000ResNet18Validator


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
    splitter = Ham10000DatasetWeightedSplitter(
        metadata_path,
        images_path,
        set_number=6,
        percent_val=0.15,
        percent_test=0.15
    )
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader


    print('3 - train model')
    NUM_BATCHES = 2
    NUM_EPOCHS_PER_BATCH = 2
    resnet18_parameters_path = base.ham10000_autoconfig.get_resnet18_parameters_path()
    timestamp = time.strftime("%Y%m%d")

    log_time('\tTraining start time:')
    tensorboard_logs = base.ham10000_autoconfig.get_tensorboard_logs_path()
    writer = SummaryWriter(log_dir=tensorboard_logs)

    running_loss = 0.0
    model = Ham10000Resnet18ModelWithDropout2d()
    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    epoch0_training_average_loss_graphicator = Ham10000ResNet18Validator(train_dataloader)
    model.eval()
    epoch0_training_average_loss_graphicator.run_epoch_validation(
        model,
        loss,
        writer,
        0,
        is_train_set=True)

    validator = Ham10000ResNet18Validator(validation_dataloader)
    model.eval()
    validator.run_epoch_validation(model, loss, writer, 0, is_train_set=False)

    log_time('\tTraining start time:')
    for current_batch in range(NUM_BATCHES):
        trained_model_filename = resnet18_parameters_path + timestamp + f'_ham10000_trained_epoch_{current_batch}.pth'

        if (current_batch > 0):
            previous_batch = current_batch - 1
            previous_trained_model_filename = resnet18_parameters_path + timestamp + f'_ham10000_trained_epoch_batch_{previous_batch}.pth'
            model.load_state_dict(torch.load(previous_trained_model_filename))

        print(f'3 - train {NUM_EPOCHS_PER_BATCH} epochs (batch {current_batch})')
        trainer = Ham10000ResNet18Trainer(train_dataloader, validation_dataloader, epochs=NUM_EPOCHS_PER_BATCH)

        tensorboard_logs = base.ham10000_autoconfig.get_tensorboard_logs_path()
        writer = SummaryWriter(log_dir=tensorboard_logs)
        print(f'current batch: {current_batch}')
        model.train()
        running_loss = trainer.run_training_and_validation_by_batch(
            model,
            loss,
            optimizer,
            writer,
            current_batch,
            NUM_EPOCHS_PER_BATCH,
            trained_model_filename,
            running_loss)

    log_time('\tTraining end time:')

    print('4 - validate model')
    validator = Ham10000ResNet18Validator(model, validation_dataloader)
    validator.run_validation()

    print('5 - make predictions')
    predictor = Ham10000ResNet18Validator(model, test_dataloader)
    predictor.run_predictor()

    writer.close()

    log_time('Done!')
