import torch


class ResNet18:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                            'resnet18',
                            pretrained=True)
        self.model.eval()
    def get_model(self):
        return self.model