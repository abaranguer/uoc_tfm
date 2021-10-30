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
            class_name = self.dataset_classes_counts.index[dataset_classe_count[0]]
            class_count = dataset_classe_count[1]
            print(
                f'\tclasse: "{class_name}"; num of images: {class_count};{(100.0 * class_count / self.num_of_images): .2f} % of the dataset.')
        print('------------------------')

        self.save_dataframe(title)

    def save_dataframe(self, title):
        dataframe_path_lnx = '/home/albert/UOC-TFM/dataframes/'
        dataframe_path_win = 'C:/albert/UOC/dataframes/'
        dataframe_path_clb = '/content/drive/MyDrive/UOC-TFM/dataframes/'
        dataframe_path = dataframe_path_win

        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = dataframe_path + timestamp + '_' + title + ".csv"

        lines = []

        num_of_images = len(self.df.values)
        for i in range(num_of_images):
            lesion_id = self.df.values[i][0]
            image_id = self.df.values[i][1]
            dx = self.df.values[i][2]
            line = f'"{lesion_id}","{image_id}","{dx}"\n'
            lines.append(line)

        with open(filename, 'w') as df_file:
            df_file.writelines(lines)

        print(f'Saved {filename}')

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
