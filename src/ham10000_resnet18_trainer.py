# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import time
import torch.optim
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from ham10000_dataset_splitter import Ham10000DatasetSplitter


class Ham10000ResNet18Trainer:

    def __init__(self, train_dataloader, model, epochs=5):
        self.train_dataloader = train_dataloader
        self.model = model
        self.epochs = epochs
        self.loss = None
        self.optimizer = None
        self.which_device = ""

    def run_training(self):
        self.loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # select device (GPU or CPU)
        self.which_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'using {self.which_device} device')
        device = torch.device(self.which_device)

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, images in enumerate(self.train_dataloader, 0):
                inputs = images['image']
                labels = images['label']

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss_current = self.loss(outputs, labels)
                loss_current.backward()
                self.optimizer.step()

                running_loss += loss_current.item()
                print(f'epoch: {epoch}; i : {i}')
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

        timestamp = time.strftime("%Y%m%d%H%M%S")
        trained_model_filename = timestamp + '_ham10000_trained_model.pth'
        torch.save(self.model.state_dict(), trained_model_filename)


if __name__ == '__main__':
    matadata_path = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    images_path = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'

    print('Start')

    splitter = Ham10000DatasetSplitter(matadata_path, images_path)
    train_dataloader = splitter.train_dataloader

    model = models.resnet18()

    trainer = Ham10000ResNet18Trainer(train_dataloader, model)
    trainer.run_training()

    print('Done!')