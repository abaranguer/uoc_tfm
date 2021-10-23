# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time
import torch
import torch.optim
import torchvision.models as models
from ham10000_dataset_splitter import Ham10000DatasetSplitter
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


class Ham10000ResNet18Trainer:

    def __init__(self, train_dataloader, model, epochs=5):
        self.train_dataloader = train_dataloader
        self.model = model
        self.epochs = epochs
        self.loss = None
        self.optimizer = None
        self.which_device = ""

    def run_training(self, writer):
        self.loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # select device (GPU or CPU)
        self.which_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'using {self.which_device} device')
        device = torch.device(self.which_device)

        num_steps = 1
        running_loss = 0.0
        num_images = 0.0

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            for i, images in enumerate(self.train_dataloader, 0):
                inputs = images['image']
                labels = images['label']
                batch_size = inputs.size(0)
                num_images += batch_size

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss_current = self.loss(outputs, labels)
                loss_current.backward()
                self.optimizer.step()

                loss_current_value = loss_current.item()
                running_loss += loss_current_value
                writer.add_scalar("loss/steps", loss_current_value, num_steps)
                writer.add_scalar("running_loss/num_images", running_loss * batch_size  / num_images, num_images)
                num_steps += 1
                print(f'epoch: {epoch}; i : {i} ')

        print('Finished Training')
        writer.flush()

        timestamp = time.strftime("%Y%m%d%H%M%S")
        trained_model_filename = timestamp + '_ham10000_trained_model.pth'
        torch.save(self.model.state_dict(), trained_model_filename)
