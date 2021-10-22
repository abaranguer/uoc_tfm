# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import pandas
import time


class Ham10000DatasetAnalyzer:
    '''
    akiec: 327
    bcc: 514
    bkl: 1099
    df: 115
    nv: 6705
    mel: 1113
    vasc: 142

    total: 10015
    '''

    def __init__(self):
        self.path = None
        self.df = None
        self.num_of_images = 0
        self.dataset_classes = 0
        self.dataset_classes_counts = None

    def analyze_path(self, path):
        self.path = path
        self.df = pandas.read_csv(path)
        self.analyze()

    def analyze_dataframe(self, df):
        self.path = None
        self.df = df
        self.analyze()

    def analyze(self):
        self.num_of_images = len(self.df['dx'])
        self.dataset_classes = self.df['dx'].unique()
        self.dataset_classes_counts = self.df['dx'].value_counts()

    def metadata(self):
        return self.num_of_images, self.dataset_classes, self.dataset_classes_counts

    def show(self, title):
        print(f'---- Analyzer. {title} ----\n')
        print(f'num of images: {self.num_of_images}')
        print(f'num of classes: {self.dataset_classes}')
        for dataset_classe_count in enumerate(self.dataset_classes_counts):
            print(
                f'\tclasse: "{self.dataset_classes[dataset_classe_count[0]]}"; num of images: {dataset_classe_count[1]};{(100.0 * dataset_classe_count[1] / self.num_of_images): .2f} % of the dataset.')
        print('------------------------')

    def save_dataframe(self, data_frame, filename):
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = timestamp + '_' + filename
        data_frame.to_pickle(filename)


if __name__ == '__main__':
    path_lnx = '/home/albert/UOC-TFM/dataset/HAM10000_metadata'
    path_win = 'C:/albert/UOC/dataset/HAM10000_metadata'
    path_clb = '/content/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata'

    analyzer = Ham10000DatasetAnalyzer()
    analyzer.analyze_path(path_win)
    num_of_images, dataset_classes, dataset_classes_counts = analyzer.metadata()

    print(f'num of images: {num_of_images}')
    print(f'num of classes: {dataset_classes}')
    for dataset_classe_count in enumerate(dataset_classes_counts):
        print(
            f'\tclasse: "{dataset_classes[dataset_classe_count[0]]}"; num of images: {dataset_classe_count[1]};{(100.0 * dataset_classe_count[1] / num_of_images): .2f} % of the dataset.')

    print('\n\nUsing "analyzer.show" method:\n')
    analyzer.show('Test on HAM10000_METADATA')
