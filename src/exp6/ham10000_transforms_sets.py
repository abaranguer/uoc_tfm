import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_set(set_number):
    if set_number == 1:
        return get_set1()
    elif set_number == 2:
        return get_set2()
    elif set_number == 3:
        return get_set3()
    elif set_number == 4:
        return get_set4()
    elif set_number == 5:
        return get_set5()
    elif set_number == 6:
        return get_set6()
    else:
        raise "Unknown set index"


def get_set1():
    return A.Compose([
        # Random crops

        A.OneOf([
            #     A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=300, scale=[0.2, 0.8], ratio=[3. / 4., 4. / 3.]),
            #     A.CenterCrop(height=135, width=180),
            #     A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=0.5),

        # Affine Transforms
        A.OneOf([
            #     A.Affine(),
            #     A.PiecewiseAffine(),
            A.ElasticTransform(),
            #     A.ShiftScaleRotate(),
            A.Compose([
                A.Rotate(),
                A.Resize(height=225, width=300)
            ]),
            #     A.SafeRotate(),
            #     A.RandomRotate90(),
            #     A.RandomScale(),
            #     A.NoOp()
        ], p=0.5),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Compose(
            #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
            # ),
            A.NoOp()
        ], p=0.5),

        # # Saturation, contrast, brightness, and hue
        # A.OneOf([
        #     A.CLAHE(),
        #     A.ColorJitter(),
        #     A.Equalize(),
        #     A.HueSaturationValue(),
        #     A.RandomBrightnessContrast(),
        #     A.NoOp()
        # ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)


def get_set2():
    return A.Compose([
        # Random crops

        A.OneOf([
            #     A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=300, scale=[0.2, 0.8], ratio=[3. / 4., 4. / 3.]),
            #     A.CenterCrop(height=135, width=180),
            #     A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=0.5),

        # Affine Transforms
        A.OneOf([
            #     A.Affine(),
            #     A.PiecewiseAffine(),
            A.ElasticTransform(),
            #     A.ShiftScaleRotate(),
            A.Compose([
                A.Rotate(),
                A.Resize(height=225, width=300)
            ]),
            #     A.SafeRotate(),
            #     A.RandomRotate90(),
            #     A.RandomScale(),
            #     A.NoOp()
        ], p=0.5),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Compose(
            #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
            # ),
            A.NoOp()
        ], p=0.5),

        # Saturation, contrast, brightness, and hue
        A.OneOf([
            #     A.CLAHE(),
            A.ColorJitter(),
            #     A.Equalize(),
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
            A.NoOp()
        ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)


def get_set3():
    return A.Compose([
        # Random crops

        A.OneOf([
            #     A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=300),
            #     A.CenterCrop(height=135, width=180),
            #     A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=0.5),

        # Affine Transforms
        A.OneOf([
            #     A.Affine(),
            #     A.PiecewiseAffine(),
            A.ElasticTransform(),
            #     A.ShiftScaleRotate(),
            A.Compose([
                A.Rotate(),
                A.Resize(height=225, width=300)
            ]),
            #     A.SafeRotate(),
            #     A.RandomRotate90(),
            #     A.RandomScale(),
            #     A.NoOp()
        ], p=0.5),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Compose(
            #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
            # ),
            A.NoOp()
        ], p=0.5),

        # Saturation, contrast, brightness, and hue
        A.OneOf([
            #     A.CLAHE(),
            #     A.ColorJitter(),
            #     A.Equalize(),
            #     A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
            A.NoOp()
        ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)


def get_set4():
    return A.Compose([
        # Random crops

        A.OneOf([
            #     A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=300),
            #     A.CenterCrop(height=135, width=180),
            #     A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=1),

        # Affine Transforms
        A.OneOf([
            #     A.Affine(),
            #     A.PiecewiseAffine(),
            A.ElasticTransform(),
            #     A.ShiftScaleRotate(),
            A.Compose([
                A.Rotate(),
                A.Resize(height=225, width=300)
            ]),
            #     A.SafeRotate(),
            #     A.RandomRotate90(),
            #     A.RandomScale(),
            #     A.NoOp()
        ], p=1),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Compose(
            #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
            # ),
            A.NoOp()
        ], p=1),

        # # Saturation, contrast, brightness, and hue
        # A.OneOf([
        #     A.CLAHE(),
        #     A.ColorJitter(),
        #     A.Equalize(),
        #     A.HueSaturationValue(),
        #     A.RandomBrightnessContrast(),
        #     A.NoOp()
        # ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)


def get_set5():
    return A.Compose([
        # Random crops

        A.OneOf([
            #     A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=300,
                                scale=[0.6, 0.8],
                                ratio=[4. / 3., 4. / 3.]),
            #     A.CenterCrop(height=135, width=180),
            #     A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=0.5),

        # Affine Transforms
        A.OneOf([
            #     A.Affine(),
            #     A.PiecewiseAffine(),
            A.ElasticTransform(),
            #     A.GridDistortion(),
            #     A.ShiftScaleRotate(),
            A.Compose([
                A.Rotate(border_mode=cv2.BORDER_CONSTANT),
                A.Resize(height=225, width=300)
            ]),
            #     A.SafeRotate(),
            #     A.RandomScale(),
            #     A.NoOp()
        ], p=0.5),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Compose(
            #     [A.Transpose(p=1), A.Rotate(limit=[90, 90], p=1)]
            # ),
            A.NoOp()
        ], p=0.5),

        # # Saturation, contrast, brightness, and hue
        # A.OneOf([
        #     A.CLAHE(),
        #     A.ColorJitter(),
        #     A.Equalize(),
        #     A.HueSaturationValue(),
        #     A.RandomBrightnessContrast(),
        #     A.NoOp()
        # ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)

def get_set6():
    return A.Compose([
        A.CenterCrop(height=225, width=225, always_apply=True),
        # Random crops

        A.OneOf([
            # A.RandomCrop(height=180, width=240),
            A.RandomResizedCrop(height=225, width=225,
                                scale=[0.6, 0.8],
                                ratio=[1., 1.]),
            A.Compose([
                A.CenterCrop(height=180, width=180),
                A.Resize(height=225, width=225)
            ]),
            # A.Crop(),
            # A.CropAndPad(percent=0.1),
            A.NoOp()
        ], p=0.5),

        # Affine Transforms
        A.OneOf([
            # A.Affine(),
            # A.PiecewiseAffine(),
            A.ElasticTransform(),
            # A.GridDistortion(),
            # A.ShiftScaleRotate(),
            # A.Compose([
            #     A.Rotate(border_mode=cv2.BORDER_CONSTANT),
            #     A.Resize(height=225, width=225)
            # ]),
            # A.SafeRotate(),
            A.RandomRotate90(), # new!
            # A.RandomScale(),
            # A.NoOp()
        ], p=0.5),

        # # Flips
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(), # new!
            A.NoOp()
        ], p=0.5),

        # # Saturation, contrast, brightness, and hue
        # A.OneOf([
        #     A.CLAHE(),
        #     A.ColorJitter(),
        #     A.Equalize(),
        #     A.HueSaturationValue(),
        #     A.RandomBrightnessContrast(),
        #     A.NoOp()
        # ], p=0.5),

        # Normalize
        A.Normalize(mean=[0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
                    std=[0.141, 0.152, 0.169],  # std. dev. of RGB channels of HAM10000 dataset
                    p=1),

        ToTensorV2()
    ], p=1)