# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torchvision
# from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import src.exp4.ham10000_autoconfig
from src.exp4.ham10000_dx_decoder import dx_to_description
from src.exp4.ham10000_dx_decoder import dx_to_int
from src.exp4.ham10000_dataset_loader import Ham10000Dataset

''' 
Albumentations:  
https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8

HAM10000
metadata header:
lesion_id, image_id, dx, dx_type, age, sex, localization, dataset

dx fields:
akiec: Actinic Keratoses i Intraepithelial Carcinoma. --  0
bcc: Basal cell carcinoma.                            --  1
bkl: "Benign keratosis".                              --  2
df: Dermatofibroma.                                   --  3
nv: Melanocytic nevi.                                 --  4
mel: Melanoma.                                        --  5
vasc: Vascular skin lesions.                          --  6
'''

class Ham10000AlbumentationDataset(Dataset):
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
        # image = Image.open(img_path).convert('RGB')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
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
    metadata_path = src.exp4.ham10000_autoconfig.get_metadata_path()
    df = pandas.read_csv(metadata_path)
    train_set, test_set = train_test_split(df, test_size=0.25)

    image_folder = src.exp4.ham10000_autoconfig.get_images_path()

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        # TODO: Utilitzar la mitjana i la desviació típica dels canals RGB de les imatges de ham10000
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Ham10000Dataset(train_set, image_folder, train_data_transform)
    test_dataset = Ham10000Dataset(test_set, image_folder, train_data_transform)
    BATCH_SIZE = 100
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
