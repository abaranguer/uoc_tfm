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
            # Estic utilitzant la mitjana i la desviació típica dels canals RGB de les imatges de ham10000 300x225
            # Valors ImageNET: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.764, 0.547, 0.571], [0.141, 0.152, 0.169])
        ])

        '''
        '# training data
        train_data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        '''

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
        classes, num_images_per_class = np.unique(unbalanced_dataset.labels[:, 1], return_counts=True)
        count_dict = dict(zip(classes, num_images_per_class))
        num_images_dataset = int(sum(num_images_per_class))

        class_weights_as_array = [(1 - (num_images / num_images_dataset)) for num_images in num_images_per_class]
        # class_weights = torch.tensor(class_weights_as_array,
        #                             dtype=torch.float)

        # class: "akiec"; num of images: 51; 3.40 % of the dataset.
        # class: "bcc"; num of images: 63; 4.19 % of the dataset.
        # class: "bkl"; num of images: 168; 11.19 % of the dataset.
        # class: "df"; num of images: 20; 1.33 % of the dataset.
        # class: "mel"; num of images: 187; 12.45 % of the dataset.
        # class: "nv"; num of images: 993; 66.11 % of the dataset.
        # class: "vasc"; num of images: 20; 1.33 % of the dataset.

        # class_weights = torch.tensor([1./0.0340, 1./0.0419, 1./0.1119, 1./0.0133, 1./0.1245, 1./0.6611, 1./0.0133],
        #                             dtype=torch.float)

        # ['akiec', 'bcc', 'bkl', 'df', 'mel' ,'nv', 'vasc']
        # class_weights = [1. / 51., 1. / 63., 1. / 168., 1. / 20., 1. / 187., 1. / 933., 1. / 20.]
        class_weights = [1502. / 51., 1502. / 63., 1502. / 168., 1502. / 20., 1502. / 187., 1502. / 933., 1502. / 20.]

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
