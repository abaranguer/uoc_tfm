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

class Ham10000ResNet18Validator:
    def __init__(self, model, splitter):
        self.model = model
        self.validation_dataloader = splitter.validation_dataloader

        self.class_names = ['', '', '', '', '', '', '']

    def run_validation(self):
        self.populate_class_names()

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
                    all_predicted.append(self.class_names[np_predicted[i]])
                    all_truth.append(self.class_names[np_label[i]])

        validation_set_confusion_matrix = sklearn.metrics.confusion_matrix(all_truth, all_predicted)
        print(validation_set_confusion_matrix)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=validation_set_confusion_matrix,
                                                      display_labels=self.class_names)
        disp.plot()
        plt.show()

        print(sklearn.metrics.classification_report(all_truth, all_predicted))

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
