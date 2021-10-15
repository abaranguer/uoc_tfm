# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import pandas

df = pandas.read_csv("/home/albert/UOC-TFM/dataset/HAM10000_metadata")

num_of_images = len(df['dx'])
dataset_classes = df['dx'].unique()

print(f'num of images: {num_of_images}')
print(f'num of classes: {dataset_classes}')

dataset_classes_counts = df['dx'].value_counts()
for dataset_classe_count in enumerate(dataset_classes_counts):
    print(f'    classe: "{dataset_classes[dataset_classe_count[0]]}"; num of images: {dataset_classe_count[1]};{(100.0 * dataset_classe_count[1] / num_of_images): .2f} % of the dataset.')

'''
LOG:

num of images: 10015
num of classes: ['bkl' 'nv' 'df' 'mel' 'vasc' 'bcc' 'akiec']
    classe: "bkl"; num of images: 6705; 66.95 % of the dataset.
    classe: "nv"; num of images: 1113; 11.11 % of the dataset.
    classe: "df"; num of images: 1099; 10.97 % of the dataset.
    classe: "mel"; num of images: 514; 5.13 % of the dataset.
    classe: "vasc"; num of images: 327; 3.27 % of the dataset.
    classe: "bcc"; num of images: 142; 1.42 % of the dataset.
    classe: "akiec"; num of images: 115; 1.15 % of the dataset.
'''