# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sklearn.metrics
import torch.optim


# from sklearn.metrics import
# confusion_matrix, multilabel_confusion_matrix, precision_recall_fscore_support,
# classification_report, ConfusionMatrixDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
# https://hungsblog.de/en/technology/how-to-calculate-mean-average-precision-map/
# https://www.analyticsvidhya.com/blog/2021/06/evaluate-your-model-metrics-for-image-classification-and-detection/
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826

class Ham1000NaiveMetrics:
    def __init__(self):
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
        self.manual_confusion_matrix = [[0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0]]
        self.total = 0
        self.correct = 0

        self.class_names = []

        self.total = 0
        self.correct = 0
        self.global_accuracy = 0.0

    def update(self, predicted_class, ground_truth_class):
        self.total += 1
        if predicted_class == ground_truth_class:
            self.correct += 1
        index_predicted_class = self.class_names.index(predicted_class)
        index_ground_truth_class = self.class_names.index(ground_truth_class)
        self.manual_confusion_matrix[index_ground_truth_class][index_predicted_class] += 1

    def show_naive_metrics(self):
        self.global_accuracy = self.correct / self.total
        print(f'Total images: {self.total}')
        print(f'Right predictions: {self.correct}')
        print(f'Global accuracy: {self.global_accuracy:.4f}')

        for row in range(0, 7):
            formatted_row = ''.join(['{:4}'.format(element)
                                     for element in self.manual_confusion_matrix[row]])
            print(f'{self.class_names[row]:>5}\t\t{formatted_row}')


class Ham10000ResNet18Validator:
    def __init__(self, model, validation_dataloader):
        self.model = model
        self.validation_dataloader = validation_dataloader

        self.class_names = ['', '', '', '', '', '', '']
        self.handmade_metrics = Ham1000NaiveMetrics()

        self.model.eval()

    def run_validation(self):
        self.populate_class_names()

        self.handmade_metrics.class_names = self.class_names
        all_predicted = []
        all_truth = []

        for i, images in enumerate(self.validation_dataloader, 0):
            inputs = images['image']
            labels = images['label']
            dx = images['dx']

            with torch.no_grad():
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                np_predicted = predicted.numpy()
                np_label = labels.numpy()
                num_elements = len(np_predicted)

                for i in range(num_elements):
                    predicted_class = self.class_names[np_predicted[i]]
                    ground_truth_class = self.class_names[np_label[i]]
                    all_predicted.append(predicted_class)
                    all_truth.append(ground_truth_class)

                    self.handmade_metrics.update(predicted_class, ground_truth_class)

        self.show_metrics(all_truth, all_predicted)

    def show_metrics(self, all_truth, all_predicted):

        print(f'\n\n Confusion matrix ("Handmade")')
        self.handmade_metrics.show_naive_metrics()


        print(f'\n\n Confusion matrix (sklearn.metrics)')
        validation_set_confusion_matrix = sklearn.metrics.confusion_matrix(all_truth,
                                                                           all_predicted,
                                                                           labels=self.class_names)
        print(validation_set_confusion_matrix)

        print(f'\n\n Multilabel Confusion Matrix ({self.class_names}) (sklearn.metrics)')
        print('TN FP')
        print('FN TP')
        validation_set_multilabel_confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(all_truth,
                                                                                                 all_predicted,
                                                                                                 labels=self.class_names)
        print(validation_set_multilabel_confusion_matrix)

        print(f'\n\n Classification report (sklearn.metrics)')
        print(sklearn.metrics.classification_report(all_truth, all_predicted, labels=self.class_names))
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=validation_set_confusion_matrix,
            display_labels=self.class_names)
        disp.plot()
        plt.show()

    def populate_class_names(self):
        for images in self.validation_dataloader:
            labels = images['label']
            dx = images['dx']

            np_labels = labels.numpy()
            num_elements = len(np_labels)

            for j in range(num_elements):
                index = np_labels[j]
                class_name = dx[j]

                self.class_names[index] = class_name

                if self.class_names.count('') == 0:
                    break
