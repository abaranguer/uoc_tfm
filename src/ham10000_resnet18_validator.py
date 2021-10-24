# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import torch.optim

# https://hungsblog.de/en/technology/how-to-calculate-mean-average-precision-map/
# https://www.analyticsvidhya.com/blog/2021/06/evaluate-your-model-metrics-for-image-classification-and-detection/
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826

class Ham10000ResNet18Validator:
    def __init__(self, model, splitter):
        self.model = model
        self.validation_dataloader = splitter.validation_dataloader

        self.correct = 0.0
        self.total = 0.0

        self.class_names = ['','','','','','','']

        self.TP_akiec = 0.0
        self.TP_bcc = 0.0
        self.TP_bkl = 0.0
        self.TP_df = 0.0
        self.TP_nv = 0.0
        self.TP_mel = 0.0
        self.TP_vasc = 0.0

        self.FP_akiec = 0.0
        self.FP_bcc = 0.0
        self.FP_bkl = 0.0
        self.FP_df = 0.0
        self.FP_nv = 0.0
        self.FP_mel = 0.0
        self.FP_vasc = 0.0

        self.FN_akiec = 0.0
        self.FN_bcc = 0.0
        self.FN_bkl = 0.0
        self.FN_df = 0.0
        self.FN_nv = 0.0
        self.FN_mel = 0.0
        self.FNP_vasc = 0.0

    def run_validation(self):
        self.correct = 0.0
        self.total = 0.0

        self.TP_akiec = 0.0
        self.TP_bcc = 0.0
        self.TP_bkl = 0.0
        self.TP_df = 0.0
        self.TP_nv = 0.0
        self.TP_mel = 0.0
        self.TP_vasc = 0.0

        self.FP_akiec = 0.0
        self.FP_bcc = 0.0
        self.FP_bkl = 0.0
        self.FP_df = 0.0
        self.FP_nv = 0.0
        self.FP_mel = 0.0
        self.FP_vasc = 0.0

        self.FN_akiec = 0.0
        self.FN_bcc = 0.0
        self.FN_bkl = 0.0
        self.FN_df = 0.0
        self.FN_nv = 0.0
        self.FN_mel = 0.0
        self.FN_vasc = 0.0

        self.num_akiec = 0.0
        self.num_bcc = 0.0
        self.num_bkl = 0.0
        self.num_df = 0.0
        self.num_nv = 0.0
        self.num_mel = 0.0
        self.num_vasc = 0.0

        self.populate_class_names()

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
                    current_dx = dx[i]
                    self.count_class_images(current_dx)

                    if (np_predicted[i] == np_label[i]):
                        self.count_true_positive_by_class(current_dx)
                    else:
                        predicted_class_name = self.class_names[np_predicted[i]]
                        true_class_name = self.class_names[np_label[i]]
                        self.count_false_positive(predicted_class_name)
                        self.count_false_negative(predicted_class_name, true_class_name)

        self.show_accuracy()
        self.show_metrics_akiec()
        self.show_metrics_bcc()
        self.show_metrics_bkl()
        self.show_metrics_df()
        self.show_metrics_nv()
        self.show_metrics_mel()
        self.show_metrics_vasc()

    def populate_class_names(self):
        #for i, images in enumerate(self.validation_dataloader, 0):
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
                    break;

    def count_false_positive(self, predicted_class_name):
        if predicted_class_name == 'akiec':
            self.FP_akiec += 1.0
        elif predicted_class_name == 'bcc':
            self.FP_bcc += 1.0
        elif predicted_class_name == 'bkl':
            self.FP_bkl += 1.0
        elif predicted_class_name == 'df':
            self.FP_df += 1.0
        elif predicted_class_name == 'nv':
            self.FP_nv += 1.0
        elif predicted_class_name == 'mel':
            self.FP_mel += 1.0
        elif predicted_class_name == 'vasc':
            self.FP_vasc += 1.0

    def count_false_negative(self, predicted_class_name, true_class_name):
        if (predicted_class_name != 'akiec') and (true_class_name == 'akiec'):
            self.FN_akiec += 1.0
        elif (predicted_class_name != 'bcc') and (true_class_name == 'bcc'):
            self.FN_bcc += 1.0
        elif (predicted_class_name != 'bkl') and (true_class_name == 'bkl'):
            self.FN_bkl += 1.0
        elif (predicted_class_name != 'df') and (true_class_name == 'df'):
            self.FN_df += 1.0
        elif (predicted_class_name != 'nv') and (true_class_name == 'nv'):
            self.FN_nv += 1.0
        elif (predicted_class_name != 'mel') and (true_class_name == 'mel'):
            self.FN_mel += 1.0
        elif (predicted_class_name != 'vasc') and (true_class_name == 'vasc'):
            self.FN_vasc += 1.0

    def count_true_positive_by_class(self, current_dx):
        if current_dx == 'akiec':
            self.TP_akiec += 1.0
        elif current_dx == 'bcc':
            self.TP_bcc += 1.0
        elif current_dx == 'bkl':
            self.TP_bkl += 1.0
        elif current_dx == 'df':
            self.TP_df += 1.0
        elif current_dx == 'nv':
            self.TP_nv += 1.0
        elif current_dx == 'mel':
            self.TP_mel += 1.0
        elif current_dx == 'vasc':
            self.TP_vasc += 1.0

        self.correct += 1.0

    def count_class_images(self, current_dx):
        if current_dx == 'akiec':
            self.num_akiec += 1.0
        elif current_dx == 'bcc':
            self.num_bcc += 1.0
        elif current_dx == 'bkl':
            self.num_bkl += 1.0
        elif current_dx == 'df':
            self.num_df += 1.0
        elif current_dx == 'nv':
            self.num_nv += 1.0
        elif current_dx == 'mel':
            self.num_mel += 1.0
        elif current_dx == 'vasc':
            self.num_vasc += 1.0

        self.total += 1.0

    def show_accuracy(self):
        self.accuracy = 100 * self.correct / self.total
        print('\n\n')
        print(f'num of correct predicted images (True positives): {self.correct}')
        print(f'num of images : {self.total}')
        print(f'Overall accuracy (OvAc): {self.accuracy: .4f}%')

    def show_metrics_akiec(self):
        self.accuracy_akiec = 100 * self.TP_akiec / self.num_akiec
        self.show_accuracy_by_class('akiec', self.accuracy_akiec,
                                    self.TP_akiec, self.FP_akiec, self.FN_akiec,
                                    self.num_akiec)

    def show_metrics_bcc(self):
        self.accuracy_bcc = 100 * self.TP_bcc / self.num_bcc
        self.show_accuracy_by_class('bcc', self.accuracy_bcc,
                                    self.TP_bcc, self.FP_bcc, self.FN_bcc,
                                    self.num_bcc)

    def show_metrics_bkl(self):
        self.accuracy_bkl = 100 * self.TP_bkl / self.num_bkl
        self.show_accuracy_by_class('bkl', self.accuracy_bkl,
                                    self.TP_bkl, self.FP_bkl, self.FN_bkl,
                                    self.num_bkl)

    def show_metrics_df(self):
        self.accuracy_df = 100 * self.TP_df / self.num_df
        self.show_accuracy_by_class('df', self.accuracy_df,
                                    self.TP_df, self.FP_df, self.FN_df,
                                    self.num_df)

    def show_metrics_nv(self):
        self.accuracy_nv = 100 * self.TP_nv / self.num_nv
        self.show_accuracy_by_class('nv', self.accuracy_nv,
                                    self.TP_nv, self.FP_nv, self.FN_nv,
                                    self.num_nv)

    def show_metrics_mel(self):
        self.accuracy_mel = 100 * self.TP_mel / self.num_mel
        self.show_accuracy_by_class('mel', self.accuracy_mel,
                                    self.TP_mel, self.FP_mel, self.FN_mel,
                                    self.num_mel)

    def show_metrics_vasc(self):
        self.accuracy_vasc = 100 * self.TP_vasc / self.num_vasc
        self.show_accuracy_by_class('vasc', self.accuracy_vasc,
                                    self.TP_vasc, self.FP_vasc, self.FN_vasc,
                                    self.num_vasc)

    def show_accuracy_by_class(self, image_class, accuracy, TP, FP, FN, num):
        print('\n')
        print(f'number of correct predictions "{image_class}" (Predicted = ground truth = "{image_class}", True positives): {TP}')
        print(f'number of wrong predictions "{image_class}" ([Predicted = "{image_class}"] != ground truth, False positives): {FP}')
        print(f'number of wrong predictions "{image_class}" (Predicted != [ground truth = "{image_class}"], False negatives): {FN}')
        print(f'num of "{image_class}" images : {num}')
        print(f'Accuracy on "{image_class}" labelled test images: {accuracy: .4f}%')
        print('----------------------------------')
