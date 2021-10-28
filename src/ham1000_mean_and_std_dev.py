from PIL import Image
import statistics
from os import listdir
from os.path import isfile, join
import time


def log_time(message):
    timestamp = time.strftime("%Y%m%d - %H%M%S")
    print(f'{message} {timestamp}')


if __name__ == '__main__':
    print('Start\n\n')

    path_base = '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/'
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
    max_value = 255.0

    mean_r_total = 0.0
    mean_g_total = 0.0
    mean_b_total = 0.0

    print('Start!')
    log_time('start time')
    for imagefile in imagefiles:
        counter += 1
        if counter % 100 == 0:
            print("Counter: ", counter)

        pil_img = Image.open(path_original_size_folder + '/' + imagefile)

        # split in RGB channels
        red_channel = pil_img.getchannel('R').getdata()
        green_channel = pil_img.getchannel('G').getdata()
        blue_channel = pil_img.getchannel('B').getdata()

        mean_r = statistics.mean(red_channel)
        mean_g = statistics.mean(green_channel)
        mean_b = statistics.mean(blue_channel)

        mean_r_total += mean_r
        mean_g_total += mean_g
        mean_b_total += mean_b

    norm_mean_r_total = mean_r_total / max_value
    norm_mean_g_total = mean_g_total / max_value
    norm_mean_b_total = mean_b_total / max_value

    print('mean red channel : ', mean_r_total)
    print('mean green channel : ', mean_g_total)
    print('mean blue channel : ', mean_b_total)

    print('normalized mean red channel : ', norm_mean_r_total)
    print('normalized mean green channel : ', norm_mean_g_total)
    print('normalized mean blue channel : ', norm_mean_b_total)

    log_time('end time')
    print('Done!')

    '''
    std_dev_r = statistics.pstdev(red_channel)
    std_dev_g = statistics.pstdev(green_channel)
    std_dev_b = statistics.pstdev(blue_channel)
    
    norm_std_dev_r = std_dev_r / mean_r
    norm_std_dev_g = std_dev_g / mean_g
    norm_std_dev_b = std_dev_b / mean_b
    
    print('std. dev. red channel : ', std_dev_r)
    print('std. dev. green channel : ', std_dev_g)
    print('std. dev. blue channel : ', std_dev_b)
    
    print('normalized std. dev. red channel : ', norm_std_dev_r)
    print('normalized std. dev. green channel : ', norm_std_dev_g)
    print('normalized std. dev. blue channel : ', norm_std_dev_b)
    '''
