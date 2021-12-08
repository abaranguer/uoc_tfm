import torch
import torch.nn as nn
import torchvision.models as models


'''
https://pytorch.org/docs/stable/generated/torch.nn.Module.html

class torch.nn.Module[source]
    Base class for all neural network modules.
    Your models should also subclass this class.
    Modules can also contain other Modules, allowing to nest them in a tree structure. 
    You can assign the submodules as regular attributes:

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

'''

class Ham10000Resnet18ModelWithDropout2d(nn.Module):
    def __init__(self):
        super(Ham10000Resnet18ModelWithDropout2d, self).__init__()
        self.drop = nn.Dropout2d(0.05)
        self.backbone = models.resnet18()

    def forward(self, x):
        x = self.backbone.conv1(x)
        '''
        https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        "Usually the input comes from nn.Conv2d modules."
        
        https://www.machinecurve.com/index.php/2021/07/07/using-dropout-with-pytorch/
        The Dropout technique can be used for avoiding overfitting in your neural network. 
        '''
        x = self.drop(x)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x
