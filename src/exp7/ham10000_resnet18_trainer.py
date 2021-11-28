# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

import src.exp7.ham10000_autoconfig
from src.exp7.ham10000_resnet18_validator import Ham10000ResNet18Validator


class Ham10000ResNet18Trainer:

    def __init__(self, train_dataloader, model, epochs=5):
        self.train_dataloader = train_dataloader
        self.model = model
        self.epochs = epochs
        self.loss = None
        self.optimizer = None
        self.which_device = ""
        self.num_akiec = 0
        self.num_bcc = 0
        self.num_bkl = 0
        self.num_df = 0
        self.num_mel = 0
        self.num_nv = 0
        self.num_vasc = 0
        self.num_total = 0

        self.model.train()

    def update_counters(self, dx):
        for current_dx in dx:
            self.num_total += 1

            if (self.num_total % 1000) == 0:
                print(f'Current : {self.num_total}')

            if current_dx == 'akiec':
                self.num_akiec += 1
            elif current_dx == 'bcc':
                self.num_bcc += 1
            elif current_dx == 'bkl':
                self.num_bkl += 1
            elif current_dx == 'df':
                self.num_df += 1
            elif current_dx == 'mel':
                self.num_mel += 1
            elif current_dx == 'nv':
                self.num_nv += 1
            elif current_dx == 'vasc':
                self.num_vasc += 1

    def show_counters(self):
        print(f'total num of images: {self.num_total}')
        print('Number o samples per class:')
        print(f'\takiec: {self.num_akiec}  ({100.0 * self.num_akiec / self.num_total:.2f} %)')
        print(f'\t  bcc: {self.num_bcc}  ({100.0 * self.num_bcc / self.num_total:.2f} %)')
        print(f'\t  bkl: {self.num_bkl}  ({100.0 * self.num_bkl / self.num_total:.2f} %)')
        print(f'\t   df: {self.num_df}  ({100.0 * self.num_df / self.num_total:.2f} %)')
        print(f'\t  mel: {self.num_mel}  ({100.0 * self.num_mel / self.num_total:.2f} %)')
        print(f'\t   nv: {self.num_nv}  ({100.0 * self.num_nv / self.num_total:.2f} %)')
        print(f'\t vasc: {self.num_vasc}  ({100.0 * self.num_vasc / self.num_total:.2f} %)')


    def run_training_and_validation(self, writer, validation_dataloader):
        self.loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # select device (GPU or CPU)
        self.which_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'using {self.which_device} device')
        device = torch.device(self.which_device)

        num_images = 0.0
        sum_loss = 0.0

        print('4 - model validation previous to model training (epoch 0)')
        graph_name = 'training - average loss vs. epoch'
        epoch0_training_average_loss_graphicator = Ham10000ResNet18Validator(self.model,
                                                                             self.train_dataloader)
        epoch0_training_average_loss_graphicator.run_epoch_validation(self.loss,
                                                                      writer,
                                                                      0,
                                                                      graph_name=graph_name)

        validator = Ham10000ResNet18Validator(self.model, validation_dataloader)
        validator.run_epoch_validation(self.loss, writer, 0)

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            for i, images_batch in enumerate(self.train_dataloader, 0):
                inputs = images_batch['image']
                labels = images_batch['label']
                dx = images_batch['dx']

                self.update_counters(dx)

                batch_size = inputs.size(0)
                num_images += batch_size
                self.display_batch(inputs, writer)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss_current = self.loss(outputs, labels)
                loss_current.backward()
                self.optimizer.step()

                loss_current_value = loss_current.item()
                sum_loss += loss_current_value
                # running_loss += loss_current_value
                # running_loss_per_train_images = running_loss / num_images

                writer.add_scalar("training - loss vs. num of training images",
                                  loss_current_value,
                                  num_images)

                # writer.add_scalar("training - running_loss/num_images",
                #                   running_loss_per_train_images,
                #                   num_images)

                print(f'epoch: {epoch + 1}; i : {i + 1} ')

            average_loss = sum_loss / num_images
            writer.add_scalar(graph_name,
                              average_loss,
                              epoch + 1)

            print(f'4 - model validation after epoch {epoch + 1}')
            validator = Ham10000ResNet18Validator(self.model, validation_dataloader)
            validator.run_epoch_validation(self.loss, writer, epoch + 1)

        self.show_counters()
        print('Finished Training')
        writer.flush()

        print(f'4 - model validation. Calculate metrics')
        validator = Ham10000ResNet18Validator(self.model, validation_dataloader)
        validator.run_validation()

        resnet18_parameters_path = src.exp7.ham10000_autoconfig.get_resnet18_parameters_path()
        timestamp = time.strftime("%Y%m%d%H%M%S")
        trained_model_filename = resnet18_parameters_path + timestamp + '_ham10000_trained_model.pth'
        torch.save(self.model.state_dict(), trained_model_filename)

    def display_batch(self, images_batch, writer):
        grid_img = torchvision.utils.make_grid(images_batch, nrow=10, normalize=True, scale_each=True)
        self.matplotlib_imshow(grid_img)
        writer.add_image('Current image batch (normalized)', grid_img)

    # see https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    def matplotlib_imshow(self, grid_img):
        np_grid_img = grid_img.numpy()
        plt.imshow(np.transpose(np_grid_img, (1, 2, 0)))
