# https://discuss.pytorch.org/t/how-does-weightedrandomsampler-work/8089
# https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from src.exp2.ham10000_analyzer import Ham10000DatasetAnalyzer
from src.exp2.ham10000_dataset_loader import Ham10000Dataset


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

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],  # mean of RGB channels of ImageNET
                [0.229, 0.224, 0.225])  # std. dev. of RGB channels of ImageNET
        ])

        self.train_dataset = Ham10000Dataset(self.train_set, dataset_images_path, self.data_transform)
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
        # final step: normalize weights
        #  mean_weights =  (W0 + W1 + W2 + ... + Wn) / NC
        #  Wi = Wi / mean_weights  ---> mean(W0,W1,W2,...,Wn) = 1
        NC = 7.0
        class_weights = num_images_dataset / (NC * num_images_per_class)
        mean_weights = sum(class_weights) / NC
        class_weights = class_weights / mean_weights

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
