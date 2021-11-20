# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

'''
Reescalar -mantenint l'aspect ratio i fent servir con interpolacio bilinear-
les imatges de ham100000 de mida original 600x450 a les següents dimensions

  60x45
 120x90
180x135
240x180
300x225
360x270
420x315
480x360
540x405

Carpeta amb el dataset original:
/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000

carpetes destí:
  60x45: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/60x45
 120x90: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/120x90
180x135: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/180x135
240x180: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/240x180
300x225: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225
360x270: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/360x270
420x315: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/420x315
480x360: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/480x360
540x405: /home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/540x405
'''

from os import listdir
from os.path import isfile, join

import cv2

print('Start\n\n')

path_base_lnx = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/'
path_base_win = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/'
path_base_clb = '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/'

path_base = path_base_win

original_size_folder = 'original-image-set_600x450'
folders = ['60x45',
           '120x90',
           '180x135',
           '240x180',
           '300x225',
           '360x270',
           '420x315',
           '480x360',
           '540x405']

path_original_size_folder = path_base + original_size_folder
print(path_original_size_folder)

imagefiles = [imagefile for imagefile in listdir(path_original_size_folder) if
              isfile(join(path_original_size_folder, imagefile))]
counter = 0
for imagefile in imagefiles:
    counter += 1

    image = cv2.imread(path_original_size_folder + '/' + imagefile)

    # https://www.pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
    # «cv2.INTER_LINEAR method performs bilinear interpolation.
    # this is the method that OpenCV uses by default when resizing images.»
    for i in range(1, 10):
        percent = 0.1 * i
        percent100 = 100.0 * percent
        image_scaled = cv2.resize(image, None, fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path_base + folders[i - 1] + '/' + imagefile, image_scaled)

    if counter % 100 == 0:
        print(f'\n\n{counter} files processed')

print('\n\nDone!')
