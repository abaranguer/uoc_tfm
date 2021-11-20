# /usr/bin/env python3.9
# -*- coding: utf-8 -*-

'''
Codi extret de
https://rwightman.github.io/pytorch-image-models/models/resnet/

@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  archivePrefix = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

To extract image features with this model, follow the timm feature extraction examples,
just change the name of the model you want to use.

How do I finetune this model?
You can finetune any of the pre-trained models just by changing the classifier (the last layer).

model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
To finetune on your own dataset, you have to write a training loop or adapt timm's training script to use your dataset.

How do I train this model?
You can follow the timm recipe scripts for training a new model afresh.
'''

import urllib

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

'''
ResNet

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, 
instead of learning unreferenced functions. 
Instead of hoping each few stacked layers directly fit a desired underlying mapping, 
residual nets let these layers fit a residual mapping. 
They stack residual blocks ontop of each other to form network: 
e.g. a ResNet-50 has fifty layers using these blocks. 
'''


class test_timm_resnet18:
    def __init__(self):
        # load a pretrained model
        # Replace the model name with the variant you want to use, e.g. resnet18. You
        #  can find the IDs in the model summaries at the top of this page.
        print('Start')
        print('Crea un backbone lleuger (un model) de tipus Resnet18, preeentrenat per a reconèixer races de gossos')
        self.model = timm.create_model('resnet18', pretrained=True)
        self.model.eval()

    def load_test_image(self, url):
        # load and preprocess the image:
        print('obté de Internet la imatge jpg d\'un gos i la desa en local')
        url, self.filename = (url, "dog.jpg")
        urllib.request.urlretrieve(url, self.filename)

    def preprocess_test_image(self):
        print('Transforma la imatge per a fer-la tractable per per la xarxa neuronal')
        print('La converteix a RGB ("descomprimeix" el JPG"')
        self.img = Image.open(self.filename).convert('RGB')
        print('la converteix a tensor')
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)
        self.tensor = transform(self.img).unsqueeze(0)  # transform and add batch dimension

    def predict_test_image(self):
        print('Analitza la imatge i fa una predicció de la raça')
        # get the model predictions:
        with torch.no_grad():
            out = self.model(self.tensor)

        print('Crea una taula amb les probabilitats de les estimacions')
        self.probabilities = torch.nn.functional.softmax(out[0], dim=0)
        print(f'Nombre de categories considerades (probabilities.shape):  {self.probabilities.shape}')
        # prints: torch.Size([1000])

    def show_results(self):
        print('Obté de la web la llista de races de gossos (la llista de classes ImageNet considerades')
        url, filename = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
        urllib.request.urlretrieve(url, filename)
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        print('Mostra la llista amb el top-5 de races més probables')
        top5_prob, top5_catid = torch.topk(self.probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
        # prints class names and probabilities like:
        # [('Samoyed', 0.6425196528434753),
        # ('Pomeranian', 0.04062102362513542),
        # ('keeshond', 0.03186424449086189),
        # ('white wolf', 0.01739676296710968),
        # ('Eskimo dog', 0.011717947199940681)]

        print('\nla imatge del gos és la d\'un samoiede.')
        print(f'\tla categoria més probable (amb p = {top5_prob[0].item()}) és la {categories[top5_catid[0]]}'),

        print('\nDone!')


if __name__ == '__main__':
    test = test_timm_resnet18()
    test.load_test_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
    test.preprocess_test_image()
    test.predict_test_image()
    test.show_results()
