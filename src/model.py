import torch
import torchvision.models as models
from torch.nn import AdaptiveMaxPool2d

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

        self.backbone = models.resnet18()
        mapper = {1: 64, 2: 128, 3: 256, 4: 512}
        self.classifier = torch.nn.Linear(mapper[num_layers], 7)
        self.num_layers = num_layers

        self.maxpool2d_test = torch.nn.AdaptiveMaxPool2d(1)

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

        # x = self.backbone.avgpool(x)    # Original . average global pool  <-- ojo
        x = self.maxpool2d_test(x)  # Output més petit.  shape = (100, 128, 5, 7)

        x = torch.flatten(x, 1)

        '''
        def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
            r"""
            Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
        
            This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
        
            Shape:
        
                - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
                  additional dimensions
                - Weight: :math:`(out\_features, in\_features)`
                - Bias: :math:`(out\_features)`
                - Output: :math:`(N, *, out\_features)`
            """
            if has_torch_function_variadic(input, weight, bias):
                return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
            return torch._C._nn.linear(input, weight, bias)     
        '''

        x = self.classifier(x)

        return x
