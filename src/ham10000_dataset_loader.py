# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

'''
https://pytorch.org/hub/pytorch_vision_resnet/
https://pytorch.org/vision/stable/models.html
https://www.programcreek.com/python/example/108007/torchvision.models.resnet18
https://towardsdatascience.com/beginners-guide-to-loading-image-data-with-pytorch-289c60b7afec
https://androidkt.com/load-custom-image-datasets-into-pytorch-dataloader-without-using-imagefolder/
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ham10000_dx_decoder import dx_to_description
from ham10000_dx_decoder import dx_to_int

'''
HAM10000
metadata header:
lesion_id, image_id, dx, dx_type, age, sex, localization, dataset

dx fied:
akiec: Actinic Keratoses i Intraepithelial Carcinoma. --  0
bcc: Basal cell carcinoma.                            --  1
bkl: "Benign keratosis".                              --  2
df: Dermatofibroma.                                   --  3
nv: Melanocytic nevi.                                 --  4
mel: Melanoma.                                        --  5
vasc: Vascular skin lesions.                          --  6
'''


class Ham10000Dataset(Dataset):
    def __init__(self, csv, img_folder, transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.image_names = self.csv[:]['image_id']
        self.labels = np.array(
            self.csv.drop(['lesion_id', 'dx_type', 'age', 'sex', 'localization', 'dataset'], axis=1))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = self.img_folder + self.image_names.iloc[index] + '.jpg'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        targets = self.labels[index]
        return {'image': image,
                'image_id': targets[0],
                'dx': targets[1],
                'label': dx_to_int[targets[1]]}


def imshow(inp, title=None):
    # imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    # lesion_id,image_id,dx,dx_type,age,sex,localization,dataset
    df = pandas.read_csv("/home/albert/UOC-TFM/dataset/HAM10000_metadata")
    train_set, test_set = train_test_split(df, test_size=0.25)

    image_folder = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/'

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Ham10000Dataset(train_set, image_folder, train_data_transform)
    test_dataset = Ham10000Dataset(test_set, image_folder, train_data_transform)
    BATCH_SIZE = 10
    TEST_BATCH_SIZE = 4

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True
    )

    images = next(iter(test_dataloader))

    output = torchvision.utils.make_grid(images['image'])

    print('Showing images:"')
    for dx, label, image_id in zip(images['dx'], images['label'], images['image_id']):
        print(f"\timage: {image_id}.jpg, dx: {dx}, {dx_to_description[dx]}: label: {label}")

    imshow(output)
