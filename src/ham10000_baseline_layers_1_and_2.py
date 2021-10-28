# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from ham10000_dataset_splitter import Ham10000DatasetSplitter
from ham10000_resnet18_predictor import Ham10000ResNet18Predictor
from ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from ham10000_resnet18_validator import Ham10000ResNet18Validator


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    metadata_path_lnx = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    metadata_path_win = 'C:/albert/UOC/dataset/HAM10000_metadata'
    metadata_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata'
    images_path_lnx = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/120x90/'
    images_path_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/120x90/'
    images_path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/120x90/'

    log_time('Start time:')

    print('1 . Splits training, validation and test sets')
    metadata_path = metadata_path_lnx
    images_path = images_path_lnx
    splitter = Ham10000DatasetSplitter(metadata_path, images_path, percent_val=0.15, percent_test=0.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    model = models.resnet18()

    # freeze all layers except 1 and 2

    '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    '''
    model.conv1.requires_grad_(True)
    model.bn1.requires_grad_(True)
    model.relu.requires_grad_(True)
    model.maxpool.requires_grad_(True)
    model.layer1.requires_grad_(True)
    model.layer2.requires_grad_(True)

    model.layer3.requires_grad_(False)
    model.layer4.requires_grad_(False)
    model.avgpool.requires_grad_(False)
    model.fc.requires_grad_(False)

    print('3 - train model')
    model.train()
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


