import math
import time
from os import listdir
from os.path import isfile, join

import ham10000_autoconfig

from PIL import Image

class Ham10000MeanAndStdDevCalculator:
    def __init__(self):
        self.max_value = 255.0

        self.mean_r = 0.0
        self.mean_g = 0.0
        self.mean_b = 0.0

        self.norm_mean_r = 0.0
        self.norm_mean_g = 0.0
        self.norm_mean_b = 0.0

        # original_size_folder = 'original-image-set_600x450'
        # folders = '60x45', '120x90', '180x135', '240x180', '300x225', '360x270', '420x315', '480x360', '300x225'
        self.path_base = ham10000_autoconfig.get_images_path()

        print(self.path_base)

        self.imagefiles = [imagefile for imagefile in listdir(self.path_base) if
                      isfile(join(self.path_base, imagefile))]

        print('Start!')
        self.mean_calculator()
        print('\n')
        self.std_dev_calculator()
        print('Done!')

    def log_time(self, message):
        timestamp = time.strftime("%Y%m%d - %H%M%S")
        print(f'{message} {timestamp}')

    def mean_calculator(self):
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0

        counter = 0.0
        image_counter = 0.0

        self.log_time('start mean calculation time')
        for imagefile in self.imagefiles:
            image_counter += 1.0
            if image_counter % 1000 == 0:
                print("Counter: ", image_counter)

            pil_img = Image.open(self.path_base + imagefile)

            # split in RGB channels
            red_channel = pil_img.getchannel('R').getdata()
            green_channel = pil_img.getchannel('G').getdata()
            blue_channel = pil_img.getchannel('B').getdata()

            for i in range(len(red_channel)):
                counter += 1.0
                sum_r += red_channel[i]
                sum_g += green_channel[i]
                sum_b += blue_channel[i]

        self.mean_r = sum_r / counter
        self.mean_g = sum_g / counter
        self.mean_b = sum_b / counter

        self.norm_mean_r = self.mean_r / self.max_value
        self.norm_mean_g = self.mean_g / self.max_value
        self.norm_mean_b = self.mean_b / self.max_value

        print(f'mean R: {self.mean_r: .3f}')
        print(f'mean G: {self.mean_g: .3f}')
        print(f'mean B: {self.mean_b: .3f}')

        print(f'normalized mean R: {self.norm_mean_r: .3f}')
        print(f'normalized mean G: {self.norm_mean_g: .3f}')
        print(f'normalized mean B: {self.norm_mean_b: .3f}')

        self.log_time('end mean calculation time')

    def std_dev_calculator(self):
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0

        norm_sum_r = 0.0
        norm_sum_g = 0.0
        norm_sum_b = 0.0

        counter = 0.0
        image_counter = 0.0

        self.log_time('start std. dev. calculation time')
        for imagefile in self.imagefiles:
            image_counter += 1.0
            if image_counter % 1000 == 0:
                print("Counter: ", image_counter)

            pil_img = Image.open(self.path_base + imagefile)

            # split in RGB channels
            red_channel = pil_img.getchannel('R').getdata()
            green_channel = pil_img.getchannel('G').getdata()
            blue_channel = pil_img.getchannel('B').getdata()

            for i in range(len(red_channel)):
                counter += 1.0
                sum_r += (red_channel[i] - self.mean_r) ** 2
                sum_g += (green_channel[i] - self.mean_g) ** 2
                sum_b += (blue_channel[i] - self.mean_b) ** 2

                norm_sum_r += ((red_channel[i] - self.mean_r) / self.max_value) ** 2
                norm_sum_g += ((green_channel[i] - self.mean_g) / self.max_value) ** 2
                norm_sum_b += ((blue_channel[i] - self.mean_b) / self.max_value) ** 2

        self.variance_r = sum_r / counter
        self.variance_g = sum_g / counter
        self.variance_b = sum_b / counter

        self.norm_variance_r = norm_sum_r / counter
        self.norm_variance_g = norm_sum_g / counter
        self.norm_variance_b = norm_sum_b / counter

        self.std_dev_r = math.sqrt(self.variance_r)
        self.std_dev_g = math.sqrt(self.variance_g)
        self.std_dev_b = math.sqrt(self.variance_b)

        self.norm_std_dev_r = math.sqrt(self.norm_variance_r)
        self.norm_std_dev_g = math.sqrt(self.norm_variance_g)
        self.norm_std_dev_b = math.sqrt(self.norm_variance_b)

        print(f'variance R: {self.variance_r: .3f}')
        print(f'variance G: {self.variance_g: .3f}')
        print(f'variance B: {self.variance_b: .3f}')

        print(f'normalized variance R: {self.norm_variance_r: .3f}')
        print(f'normalized variance G: {self.norm_variance_g: .3f}')
        print(f'normalized variance B: {self.norm_variance_b: .3f}')

        print(f'std. dev. R: {self.std_dev_r: .3f}')
        print(f'std. dev. G: {self.std_dev_g: .3f}')
        print(f'std. dev. B: {self.std_dev_b: .3f}')

        print(f'normalized std. dev. R: {self.norm_std_dev_r: .3f}')
        print(f'normalized std. dev. G: {self.norm_std_dev_g: .3f}')
        print(f'normalized std. dev. B: {self.norm_std_dev_b: .3f}')

        self.log_time('end std. dev. calculation time')

if __name__ == "__main__":
    calculator = Ham10000MeanAndStdDevCalculator()

'''
mean R:  194.793
mean G:  139.391
mean B:  145.612

normalized mean R:  0.764
normalized mean G:  0.547
normalized mean B:  0.571


variance R:  1292.063
variance G:  1510.033
variance B:  1866.938

normalized variance R:  0.020
normalized variance G:  0.023
normalized variance B:  0.029

std. dev. R:  35.945
std. dev. G:  38.859
std. dev. B:  43.208

normalized std. dev. R: 0.141
normalized std. dev. G: 0.152
normalized std. dev. B: 0.169
'''