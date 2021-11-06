# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

from torch.utils.tensorboard import SummaryWriter

import ham10000_autoconfig
from ham10000_dataset_splitter import Ham10000DatasetSplitter
from ham10000_resnet18_predictor import Ham10000ResNet18Predictor
from ham10000_resnet18_trainer import Ham10000ResNet18Trainer
from ham10000_resnet18_validator import Ham10000ResNet18Validator
from model import Model


def log_time(message):
    start_time = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {start_time}')


if __name__ == '__main__':
    log_time('Start time:')

    print('1 . Splits training, validation and test sets')
    metadata_path = ham10000_autoconfig.get_metadata_path()
    images_path = ham10000_autoconfig.get_images_path()
    splitter = Ham10000DatasetSplitter(metadata_path, images_path, percent_val=0.15, percent_test=0.15)
    train_dataloader = splitter.train_dataloader
    validation_dataloader = splitter.validation_dataloader
    test_dataloader = splitter.test_dataloader

    print('2 - create ResNet18 model')
    #model = models.resnet18()
    model_2_layers = Model(2)

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

    # aquí hacemos forward

    print('3 - train model')
    model_2_layers.train()
    trainer = Ham10000ResNet18Trainer(train_dataloader, model_2_layers, epochs=5)

    log_time('\tTraining start time:')
    tensorboard_logs_path = ham10000_autoconfig.get_tensorboard_logs_path()
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
