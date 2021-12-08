# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sklearn.metrics
import torch.optim
import torchvision.transforms as T


class Ham1000NaiveMetrics:
    def __init__(self):
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
        print(f'Global accuracy: {self.global_accuracy:.2f}')

        for row in range(0, 7):
            formatted_row = ''.join(['{:4}'.format(element)
                                     for element in self.manual_confusion_matrix[row]])
            print(f'{self.class_names[row]:>5}\t\t{formatted_row}')


class Ham10000ResNet18Validator:
    def __init__(self, validation_dataloader):
        self.validation_dataloader = validation_dataloader

        self.class_names = ['', '', '', '', '', '', '']

        self.all_predicted_scores = []  # ndarray of shape (n_samples,) or (n_samples, n_classes)
        self.all_predicted_labels = []  # akiec, bcc, bkl, df, mel, nv, vasc
        self.all_predicted_labels_indexes = []  # 0..6

        self.all_ground_truth_scores = []  # ndarray of shape (n_samples,) or (n_samples, n_classes)
        self.all_ground_truth_labels = []  # akiec, bcc, bkl, df, mel, nv, vasc
        self.all_ground_truth_indexes = []  # 0..6

        self.all_predicted_scores_akiec = []
        self.all_ground_truth_akiec = []
        self.all_predicted_scores_bcc = []
        self.all_ground_truth_bcc = []
        self.all_predicted_scores_bkl = []
        self.all_ground_truth_bkl = []
        self.all_predicted_scores_df = []
        self.all_ground_truth_df = []
        self.all_predicted_scores_mel = []
        self.all_ground_truth_mel = []
        self.all_predicted_scores_nv = []
        self.all_ground_truth_nv = []
        self.all_predicted_scores_vasc = []
        self.all_ground_truth_vasc = []

        self.mAP_akiec = 0.0
        self.mAP_bcc = 0.0
        self.mAP_bkl = 0.0
        self.mAP_df = 0.0
        self.mAP_mel = 0.0
        self.mAP_nv = 0.0
        self.mAP_vasc = 0.0

        self.mAP_micro = 0.0
        self.mAP_macro = 0.0
        self.mAP_weighted = 0.0

        self.handmade_metrics = Ham1000NaiveMetrics()

        self.augmented = T.Compose([
            T.CenterCrop(size=(225, 225)),
            T.TenCrop(size=[180, 180]),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
            T.Lambda(lambda crops: torch.stack([T.Resize(size=[225, 225])(crop) for crop in crops])),
            T.Normalize(
                [0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                [0.141, 0.152, 0.169])  # std. dev. of RGB channels of HAM10000 dataset
        ])

    def run_epoch_validation(self, model, loss, writer, epoch, graph_name='validation - average loss vs. epoch'):
        # self.populate_class_names()
        # self.handmade_metrics.class_names = self.class_names

        sum_loss = 0.0
        num_images = 0.0

        for i, images in enumerate(self.validation_dataloader, 0):
            inputs = images['image']
            labels = images['label']
            batch_size = inputs.size(0)
            num_images += batch_size

            with torch.no_grad():
                outputs = model(inputs)
                loss_current = loss(outputs, labels)

                loss_current_value = loss_current.item()
                sum_loss += loss_current_value
                # running_loss_per_train_images = running_loss / num_images

        average_loss = sum_loss / num_images
        writer.add_scalar(graph_name,
                          average_loss,
                          epoch)

        # writer.add_scalar("validation - running_loss/num_images",
        #                   running_loss_per_train_images,
        #                   epoch)

    def run_validation(self, model):
        self.populate_class_names()

        self.handmade_metrics.class_names = self.class_names

        for i, images in enumerate(self.validation_dataloader, 0):
            inputs = images['image']
            labels = images['label']

            with torch.no_grad():
                outputs = model(inputs)
                _, predicted_label = torch.max(outputs.data, 1)
                self.precalculate_metrics(outputs, labels)
                self.calculate_handmade_metrics(predicted_label, labels)

        self.show_mAP()
        self.show_metrics()

    def run_tta_validation(self, model):
        self.populate_class_names()

        self.handmade_metrics.class_names = self.class_names

        for i, images in enumerate(self.validation_dataloader, 0):
            if i % 10 == 0:
                print(f'validation iteration {i}')
            inputs = images['image']
            labels = images['label']

            with torch.no_grad():
                outputs = model(inputs)

                j = 0
                for image in inputs:
                    pil_image = T.ToPILImage()(image.squeeze_(0))
                    augmented_images = self.augmented(pil_image)
                    outputs_tta = model(augmented_images)
                    outputs_tta_plus_original = torch.vstack([outputs_tta,
                                                              outputs[j]])
                    outputs[j].data = torch.mean(outputs_tta_plus_original.data, 0)
                    j += 1

                _, predicted_label = torch.max(outputs.data, 1)
                self.precalculate_metrics(outputs, labels)
                self.calculate_handmade_metrics(predicted_label, labels)

        self.show_mAP()
        self.show_metrics()

    def precalculate_metrics(self, outputs, labels):
        predicted = outputs.data[:, :7]

        ix_akiec = self.class_names.index('akiec')
        ix_bcc = self.class_names.index('bcc')
        ix_bkl = self.class_names.index('bkl')
        ix_df = self.class_names.index('df')
        ix_nv = self.class_names.index('nv')
        ix_mel = self.class_names.index('mel')
        ix_vasc = self.class_names.index('vasc')

        for pred_row in predicted:
            normalized_pred_row = torch.softmax(pred_row, dim=0)

            self.all_predicted_scores.append(normalized_pred_row.numpy())

            self.all_predicted_scores_akiec.append(float(normalized_pred_row[ix_akiec]))
            self.all_predicted_scores_bcc.append(float(normalized_pred_row[ix_bcc]))
            self.all_predicted_scores_bkl.append(float(normalized_pred_row[ix_bkl]))
            self.all_predicted_scores_df.append(float(normalized_pred_row[ix_df]))
            self.all_predicted_scores_mel.append(float(normalized_pred_row[ix_mel]))
            self.all_predicted_scores_nv.append(float(normalized_pred_row[ix_nv]))
            self.all_predicted_scores_vasc.append(float(normalized_pred_row[ix_vasc]))

        for index_label in labels:
            ground_truth = [0, 0, 0, 0, 0, 0, 0]
            ground_truth[index_label] = 1

            self.all_ground_truth_scores.append(ground_truth)

            self.all_ground_truth_akiec.append(ground_truth[ix_akiec])
            self.all_ground_truth_bcc.append(ground_truth[ix_bcc])
            self.all_ground_truth_bkl.append(ground_truth[ix_bkl])
            self.all_ground_truth_df.append(ground_truth[ix_df])
            self.all_ground_truth_mel.append(ground_truth[ix_mel])
            self.all_ground_truth_nv.append(ground_truth[ix_nv])
            self.all_ground_truth_vasc.append(ground_truth[ix_vasc])

    def calculate_mAP_per_class(self):
        self.mAP_akiec = sklearn.metrics.average_precision_score(self.all_ground_truth_akiec,
                                                                 self.all_predicted_scores_akiec)
        self.mAP_bcc = sklearn.metrics.average_precision_score(self.all_ground_truth_bcc,
                                                               self.all_predicted_scores_bcc)
        self.mAP_bkl = sklearn.metrics.average_precision_score(self.all_ground_truth_bkl,
                                                               self.all_predicted_scores_bkl)
        self.mAP_df = sklearn.metrics.average_precision_score(self.all_ground_truth_df,
                                                              self.all_predicted_scores_df)
        self.mAP_mel = sklearn.metrics.average_precision_score(self.all_ground_truth_mel,
                                                               self.all_predicted_scores_mel)
        self.mAP_nv = sklearn.metrics.average_precision_score(self.all_ground_truth_nv,
                                                              self.all_predicted_scores_nv)
        self.mAP_vasc = sklearn.metrics.average_precision_score(self.all_ground_truth_vasc,
                                                                self.all_predicted_scores_vasc)

        self.mAP_micro = sklearn.metrics.average_precision_score(self.all_ground_truth_scores,
                                                                 self.all_predicted_scores,
                                                                 average='micro')
        self.mAP_macro = sklearn.metrics.average_precision_score(self.all_ground_truth_scores,
                                                                 self.all_predicted_scores,
                                                                 average='macro')
        self.mAP_weighted = sklearn.metrics.average_precision_score(self.all_ground_truth_scores,
                                                                    self.all_predicted_scores,
                                                                    average='weighted')

    def calculate_handmade_metrics(self, predicted_label, labels):
        np_predicted = predicted_label.numpy()
        np_label = labels.numpy()
        num_elements = len(np_predicted)

        for i in range(num_elements):
            predicted_class = self.class_names[np_predicted[i]]
            ground_truth_class = self.class_names[np_label[i]]
            self.all_predicted_labels.append(predicted_class)
            self.all_predicted_labels_indexes.append(np_predicted[i])
            self.all_ground_truth_labels.append(ground_truth_class)
            self.all_ground_truth_indexes.append(np_label[i])

            self.handmade_metrics.update(predicted_class, ground_truth_class)

    def show_mAP(self):
        self.calculate_mAP_per_class()
        print('----------------------------------------------------')
        print(f'mAP akiec:  {self.mAP_akiec: .2f} sobre un total de {sum(self.all_ground_truth_akiec)}')
        print(f'mAP bcc:    {self.mAP_bcc: .2f} sobre un total de {sum(self.all_ground_truth_bcc)}')
        print(f'mAP bkl:    {self.mAP_bkl: .2f} sobre un total de {sum(self.all_ground_truth_bkl)}')
        print(f'mAP df:     {self.mAP_df: .2f} sobre un total de {sum(self.all_ground_truth_df)}')
        print(f'mAP mel:    {self.mAP_mel: .2f} sobre un total de {sum(self.all_ground_truth_mel)}')
        print(f'mAP nv:     {self.mAP_nv: .2f} sobre un total de {sum(self.all_ground_truth_nv)}')
        print(f'mAP vasc:   {self.mAP_vasc: .2f} sobre un total de {sum(self.all_ground_truth_vasc)}')
        print('\n')
        print(f'mAP micro:    {self.mAP_micro: .2f}')
        print(f'mAP macro:    {self.mAP_macro: .2f}')
        print(f'mAP weighted: {self.mAP_weighted: .2f}')

        print('----------------------------------------------------')

    def show_metrics(self):
        print(f'\n\n Confusion matrix ("Handmade")')
        self.handmade_metrics.show_naive_metrics()

        print(f'\n\n Confusion matrix (sklearn.metrics)')
        validation_set_confusion_matrix = sklearn.metrics.confusion_matrix(self.all_ground_truth_labels,
                                                                           self.all_predicted_labels,
                                                                           labels=self.class_names)
        print(validation_set_confusion_matrix)

        print(f'\n\n Multilabel Confusion Matrix ({self.class_names}) (sklearn.metrics)')
        print('TN FP')
        print('FN TP')
        validation_set_multilabel_confusion_matrix = sklearn.metrics.multilabel_confusion_matrix(
            self.all_ground_truth_labels,
            self.all_predicted_labels,
            labels=self.class_names
        )

        print(validation_set_multilabel_confusion_matrix)

        print(f'\n\n Classification report (sklearn.metrics)')
        print(sklearn.metrics.classification_report(self.all_ground_truth_labels,
                                                    self.all_predicted_labels,
                                                    labels=self.class_names))
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
