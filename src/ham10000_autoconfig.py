import os

ham10000_os_config = {
    "metadata_path_lnx": '/home/albert/UOC-TFM/dataset/HAM10000_metadata',
    "metadata_path_win": 'C:/albert/UOC/dataset/HAM10000_metadata',
    "metadata_path_clb": '/content/drive/MyDrive/UOC-TFM/dataset/HAM10000_metadata',

    "images_path_lnx": '/home/albert/UOC-TFM/dataset/dataset ham_10000/ham10000/300x225/',
    "images_path_win": 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225/',
    "images_path_clb": '/content/drive/MyDrive/UOC-TFM/dataset/dataset_ham_10000/ham10000/300x225/',

    "resnet18_parameters_path_win": 'C:/albert/UOC/resnet18_parameters/',
    "resnet18_parameters_path_lnx": '/home/albert/UOC-TFM/resnet18_parameters/',
    "resnet18_parameters_path_clb": '/content/drive/MyDrive/UOC-TFM/d/resnet18_parameters/',

    "dataframe_path_lnx": '/home/albert/UOC-TFM/dataframes/',
    "dataframe_path_win": 'C:/albert/UOC/dataframes/',
    "dataframe_path_clb": '/content/drive/MyDrive/UOC-TFM/dataframes/'
}


def get_os_suffix():
    os_type = '_lnx'

    if is_colab():
        os_type = '_clb'
    elif is_win():
        os_type = '_win'

    return os_type


def is_colab():
    try:
        import google.colab
        in_colab = True
    except:
        in_colab = False

    return in_colab


def is_win():
    in_win = False

    if os.name == 'nt':
        in_win = True

    return in_win


def get_metadata_path():
    key = 'metadata_path' + get_os_suffix()
    return ham10000_os_config[key]


def get_images_path():
    key = 'images_path' + get_os_suffix()
    return ham10000_os_config[key]


def get_resnet18_parameters_path():
    key = 'resnet18_parameters_path' + get_os_suffix()
    return ham10000_os_config[key]


def get_dataframe_path():
    key = 'dataframe_path' + get_os_suffix()
    return ham10000_os_config[key]
