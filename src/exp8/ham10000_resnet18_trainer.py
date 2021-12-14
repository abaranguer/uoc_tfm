# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torchvision


class Ham10000ResNet18Trainer:

    def __init__(self, train_dataloader, epochs=5):
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.which_device = ""
        self.num_akiec = 0
        self.num_bcc = 0
        self.num_bkl = 0
        self.num_df = 0
        self.num_mel = 0
        self.num_nv = 0
        self.num_vasc = 0
        self.num_total = 0

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
        print('Number of samples per class:')
        print(f'\takiec: {self.num_akiec}  ({100.0 * self.num_akiec / self.num_total:.2f} %)')
        print(f'\t  bcc: {self.num_bcc}  ({100.0 * self.num_bcc / self.num_total:.2f} %)')
        print(f'\t  bkl: {self.num_bkl}  ({100.0 * self.num_bkl / self.num_total:.2f} %)')
        print(f'\t   df: {self.num_df}  ({100.0 * self.num_df / self.num_total:.2f} %)')
        print(f'\t  mel: {self.num_mel}  ({100.0 * self.num_mel / self.num_total:.2f} %)')
        print(f'\t   nv: {self.num_nv}  ({100.0 * self.num_nv / self.num_total:.2f} %)')
        print(f'\t vasc: {self.num_vasc}  ({100.0 * self.num_vasc / self.num_total:.2f} %)')

    def run_training_by_epoch(self, model, loss, optimizer, writer, epoch, trained_model_filename, running_loss):
        # select device (GPU or CPU)
        self.which_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'using {self.which_device} device')
        device = torch.device(self.which_device)

        num_steps = 1
        num_images = epoch * len(self.train_dataloader.dataset)

        for i, images_batch in enumerate(self.train_dataloader, 0):
            inputs = images_batch['image']
            labels = images_batch['label']
            dx = images_batch['dx']

            self.update_counters(dx)

            batch_size = inputs.size(0)
            num_images += batch_size
            self.display_batch(inputs, writer)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss_current = loss(outputs, labels)
            loss_current.backward()
            optimizer.step()

            loss_current_value = loss_current.item()
            running_loss += loss_current_value
            running_loss_per_train_images = running_loss / num_images

            writer.add_scalar(f"running_loss/num_images", running_loss_per_train_images, num_images)

            num_steps += 1
            print(f'epoch: {epoch}; i : {i} ')

        self.show_counters()
        print('Finished Training')

        torch.save(model.state_dict(), trained_model_filename)

        return running_loss

    def display_batch(self, images_batch, writer):
        grid_img = torchvision.utils.make_grid(images_batch, nrow=10, normalize=True, scale_each=True)
        self.matplotlib_imshow(grid_img)
        writer.add_image('Current image batch (normalized)', grid_img)

    def matplotlib_imshow(self, grid_img):
        np_grid_img = grid_img.numpy()
        plt.imshow(np.transpose(np_grid_img, (1, 2, 0)))
