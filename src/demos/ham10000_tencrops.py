import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.utils as U



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


torch.manual_seed(42)
img_path = 'C:/albert/UOC/dataset/dataset ham_10000/ham10000/300x225/ISIC_0024306.jpg'
image_pil_1 = PIL.Image.open(img_path).convert('RGB')

augmented = T.Compose([
    T.TenCrop(size=[180,240]),
    T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
    T.Lambda(lambda crops: torch.stack([T.Resize(size=[225,300])(crop) for crop in crops])),
    T.Normalize(
        [0.764, 0.547, 0.571],  # mean of RGB channels of HAM10000 dataset
        [0.141, 0.152, 0.169])  # std. dev. of RGB channels of HAM10000 dataset
])

augmented_images = augmented(image_pil_1)

image_grid = U.make_grid(
    augmented_images,
    nrow=10,
    normalize=True,
    scale_each=True)

plt.imshow(image_grid.permute(1, 2, 0))
plt.show()
