import PIL
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2


def view_transform(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


albumentation_transforms = A.Compose([
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

img_path = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225/ISIC_0024306.jpg'
image_pil_1 = PIL.Image.open(img_path).convert('RGB')
image_np_1 = np.array(image_pil_1).astype(float)
image_tensor_1 = torch.from_numpy(image_np_1)
# plt.imshow(image_pil_1)
# plt.show()
images = [image_np_1]
for i in range(99):
    augmented = albumentation_transforms(image=image_np_1)
    toPilImage = torchvision.transforms.ToPILImage()
    image_pil_2 = toPilImage(augmented['image']).convert('RGB')
    image_np_2 = np.array(image_pil_2).astype(float)
    image_tensor_2 = torch.from_numpy(image_np_2)
    # plt.imshow(image_pil_2)
    # plt.show()
    images.append(image_np_2)

grid_tensor = torch.from_numpy(np.array(images))
# (N, C, H, W)
reordered_grid_tensor = torch.permute(grid_tensor, [0, 3, 1, 2])
image_grid = torchvision.utils.make_grid(reordered_grid_tensor,
                                         nrow=10,
                                         normalize=True,
                                         scale_each=True)
plt.imshow(image_grid.permute(1, 2, 0))
plt.show()
