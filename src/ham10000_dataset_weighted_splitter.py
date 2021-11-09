# https://discuss.pytorch.org/t/how-does-weightedrandomsampler-work/8089
# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from ham10000_analyzer import Ham10000DatasetAnalyzer
from ham10000_dataset_loader import Ham10000Dataset
from ham10000_albumentation_dataset_loader import Ham10000AlbumentationDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class Ham10000DatasetWeightedSplitter:

    def __init__(self, dataset_metadata_path, dataset_images_path,
                 percent_val=0.15, percent_test=0.15,
                 BATCH_SIZE=100, VAL_BATCH_SIZE=20, TEST_BATCH_SIZE=20):
        np.random.seed(0)
        self.analyzer = Ham10000DatasetAnalyzer()
        self.analyzer.analyze_path(dataset_metadata_path)
        self.analyzer.show('FULL DATASET')

        df = pandas.read_csv(dataset_metadata_path)
        percent_validation = percent_val + percent_test

        # 42? Don't panic! Read the "Hitchhikers guide to galaxy"!
        self.train_set, val_test_set = train_test_split(
            df,
            test_size=percent_validation,
            random_state=42
        )

        percent_test_validation = percent_test / percent_validation
        self.validation_set, self.test_set = train_test_split(val_test_set, test_size=percent_test_validation)

        self.analyzer.analyze_dataframe(self.train_set)
        self.analyzer.show('TRAIN SET')

        self.analyzer.analyze_dataframe(self.validation_set)
        self.analyzer.show('VALIDATION SET')

        self.analyzer.analyze_dataframe(self.test_set)
        self.analyzer.show('TEST SET')

        # data augmentation
        # pip install -U albumentations
        # https://albumentations.ai/
        # @Article{info11020125,
        #     AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
        #     TITLE = {Albumentations: Fast and Flexible Image Augmentations},
        #     JOURNAL = {Information},
        #     VOLUME = {11},
        #     YEAR = {2020},
        #     NUMBER = {2},
        #     ARTICLE-NUMBER = {125},
        #     URL = {https://www.mdpi.com/2078-2489/11/2/125},
        #     ISSN = {2078-2489},
        #     DOI = {10.3390/info11020125}
        # }
        #
        # https://albumentations.ai/docs/examples/pytorch_classification/
        # Be cautious when using data augemtation! Read
        # https://towardsdatascience.com/data-augmentation-in-medical-images-95c774e6eaae
        # Data Augmentation in Medical Images
        # How to improve vision model performance by reshaping and resampling data
        # Cody Glickman, PhD  Oct 12, 2020. (last seen in Nov. 10th, 2021).
        # "When performing any type of data augmentation, it is important to keep in mind
        # the output of your model and if augmentation would affect the resulting classification.
        # For example, in X-ray data the heart is typically on the right of the image, however
        # the image below shows a horizontal flip augmentation inadvertently creates a medical
        # condition call situs inversus."

        # Estic utilitzant la mitjana i la desviació típica dels canals RGB de les imatges de ham10000 300x225
        # Valors ImageNET: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.albumentation_transforms = A.Compose([
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(),
                A.GaussNoise(),
            ], p=0.5),
            A.Normalize(
                mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                std=[0.141, 0.152, 0.169]),  # std. dev. of RGB channels of HAM10000 dataset
            ToTensorV2()
            ], p=1
        )

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                [0.141, 0.152, 0.169])  # std. dev. of RGB channels of HAM10000 dataset
        ])

        '''
        train_data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],  # mean of RGB channels of ImageNet dataset 
                [0.229, 0.224, 0.225])  # std. dev. of RGB channels of ImageNet dataset
        ])
        '''

        self.train_dataset = Ham10000AlbumentationDataset(self.train_set, dataset_images_path, self.albumentation_transforms)
        self.validation_dataset = Ham10000Dataset(self.validation_set, dataset_images_path, self.data_transform)
        self.test_dataset = Ham10000Dataset(self.test_set, dataset_images_path, self.data_transform)

        weighted_sampler = self.weighted_sampler_factory(self.train_dataset)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=weighted_sampler
        )

        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False
        )

    def weighted_sampler_factory(self, unbalanced_dataset):
        _, num_images_per_class = np.unique(unbalanced_dataset.labels[:, 1], return_counts=True)
        num_images_dataset = int(sum(num_images_per_class))

        # how to calculate the class weights
        # let
        #   Wi = Weight of class i
        #   NI = number of images (7010)
        #   NO = number of images in sample set (i.e. 10515 = 1010 * 1.5)
        #   NC = number of classes (7)
        #   NSi = number of images of class i
        #   NS0 = number of samples of class "akiec" (231).
        #   NS1 = number of samples of class "bcc"   (371)
        #   ...
        #   NOi = number of images of class i in sample set ( aprox. 1502 = NO / NC )
        #   Wi  = "weight" or "intensity" of sampling in class i
        # then
        #   NSi * Wi = NO / NC
        # it implies
        #   Wi = NO / (NC * NSi)
        NC = 7.0
        class_weights = num_images_dataset / (NC * num_images_per_class)

        sample_weights = [0] * num_images_dataset

        index_dx = 0
        for current_dx in unbalanced_dataset.labels[:, 1]:
            if current_dx == 'akiec':
                sample_weights[index_dx] = class_weights[0]
            elif current_dx == 'bcc':
                sample_weights[index_dx] = class_weights[1]
            elif current_dx == 'bkl':
                sample_weights[index_dx] = class_weights[2]
            elif current_dx == 'df':
                sample_weights[index_dx] = class_weights[3]
            elif current_dx == 'mel':
                sample_weights[index_dx] = class_weights[4]
            elif current_dx == 'nv':
                sample_weights[index_dx] = class_weights[5]
            elif current_dx == 'vasc':
                sample_weights[index_dx] = class_weights[6]
            index_dx += 1

        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_images_dataset,
            replacement=True
        )

        return weighted_sampler
