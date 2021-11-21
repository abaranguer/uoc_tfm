# https://discuss.pytorch.org/t/how-does-weightedrandomsampler-work/8089
# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from src.exp4.ham10000_analyzer import Ham10000DatasetAnalyzer
from src.exp4.ham10000_dataset_loader import Ham10000Dataset
from src.exp4.ham10000_albumentation_dataset_loader import Ham10000AlbumentationDataset

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
        # https://albumentations.ai/docs/examples/pytorch_classification/
        # Be cautious when using data augemtation! Read
        # https://towardsdatascience.com/data-augmentation-in-medical-images-95c774e6eaae

        self.albumentation_transforms = A.Compose([
            # Random crops

            A.OneOf([
                #     A.RandomCrop(height=180, width=240),
                A.RandomResizedCrop(height=225, width=300, scale=[0.2, 0.8], ratio=[3. / 4., 4. / 3.]),
                #     A.CenterCrop(height=135, width=180),
                #     A.Crop(),
                # A.CropAndPad(percent=0.1),
                A.NoOp()
            ], p=0.5),

            # Affine Transforms
            A.OneOf([
                #     A.Affine(),
                #     A.PiecewiseAffine(),
                A.ElasticTransform(),
                #     A.ShiftScaleRotate(),
                A.Compose([
                    A.Rotate(),
                    A.Resize(height=225, width=300)
                ]),
                #     A.SafeRotate(),
                #     A.RandomRotate90(),
                #     A.RandomScale(),
                #     A.NoOp()
            ], p=0.5),

            # # Flips
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # A.Compose(
                #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
                # ),
                A.NoOp()
            ], p=0.5),

            # # Saturation, contrast, brightness, and hue
            # A.OneOf([
            #     A.CLAHE(),
            #     A.ColorJitter(),
            #     A.Equalize(),
            #     A.HueSaturationValue(),
            #     A.RandomBrightnessContrast(),
            #     A.NoOp()
            # ], p=0.5),

            # Normalize
            A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                        std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                        p=1),

            ToTensorV2()
        ], p=1)

        '''
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],  # mean of RGB channels of ImageNET dataset
                [0.229, 0.224, 0.225])  # std. dev. of RGB channels of ImageNET dataset
        ])
        '''

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                [0.141, 0.152, 0.169])  # std. dev. of RGB channels of HAM10000 dataset
        ])

        self.train_dataset = Ham10000AlbumentationDataset(self.train_set, dataset_images_path, self.albumentation_transforms)
        # self.train_dataset = Ham10000Dataset(self.train_set, dataset_images_path, self.data_transform)
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
        #   NI = number of images full set (10015)
        #   NO = number of images in train set (7010)
        #   NC = number of classes  (7)
        #   NSi = number of images of class i
        #     NS0 = number of samples of class "akiec"  (231)
        #     NS1 = number of samples of class "bcc" (371)
        #     NS2 = number of samples of class "bkl"  (749)
        #     NS3 = number of samples of class "df" (76)
        #     NS4 = number of samples of class "mel" (766)
        #     NS5 = number of samples of class "nv"  (4708)
        #     NS6 = number of samples of class "vasc" (109)
        #   NOi = number of images of class i in train  set
        #   Wi  = "weight" or "intensity" of sampling train set for class i
        # then
        #   NSi * Wi = NO / NC
        # it implies
        #   Wi = NO / (NC * NSi)
        #     W0 = 4.36
        #     W1 = 2.7
        #     W2 = 1.34
        #     W3 = 13.18
        #     W4 = 1.31
        #     W5 = 0.21
        #     W6 = 9.19
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
