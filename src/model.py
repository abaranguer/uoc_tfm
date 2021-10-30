import torchvision.models as models
import torch


def freeze_layer(module):
    for name, param in module.named_parameters():
        param.requires_grad = False
    module.apply(freeze_bn)


def freeze_bn(module):
    if isinstance(module, (torch.nn.BatchNorm1d,
                           torch.nn.BatchNorm2d,
                           torch.nn.BatchNorm3d)):
        module.eval()

class Model(torch.nn.Module):
    def __init__(self, num_layers):
        super(Model, self).__init__()

        self.backbone =  models.resnet18()
        mapper = {1:64, 2:128, 3:256, 4:512}
        self.classifier = torch.nn.Linear(mapper[num_layers], 7)
        self.num_layers = num_layers

        for x in range(num_layers + 1, 5):
            self.backbone.__delattr__(f"layer{x}")

    # freeze layers
    def freeze_layers(self, num_layers):
        modules = [self.backbone.conv1, self.backbone.bn1, self.backbone.relu]

        for x in range(num_layers):
            modules += [self.backbone.__getattr__(f"layer{x}")]

        for module in modules:
            module.apply(freeze_layer)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        for i in range(1, self.num_layers + 1):
            x = self.backbone.__getattr__(f"layer{i}")(x)

        x = self.backbone.avgpool(x)    #  TODO. Output más pequeño.
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # TODO
        '''
        imágenes más pequeñas
        maxpool x avgpool
        añadir algo por el estilo de self.maxpool = torch.nn.MaxPool()
        o torch.nn.functional.max_pool
        '''


        return x

