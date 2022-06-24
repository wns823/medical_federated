import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from efficientnet_pytorch import EfficientNet
from efficientnet_gn import EfficientNet_GN


##############################################################################################################

class SkinModel(nn.Module):
    def __init__(self, emb_type, pretrain=False, n_class=8):
        super(SkinModel, self).__init__()
        self.emb_type = emb_type
        self.pretrain = pretrain
        
        if emb_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.emb_dim = list(model.children())[-1].in_features
            self.linear = nn.Linear(in_features=self.emb_dim, out_features=n_class, bias=True)
        elif emb_type == "vgg16_bn" :
            self.encoder = models.vgg16_bn(pretrained=True)
            num_ftrs = self.encoder.classifier[-1].in_features
            self.encoder.classifier[-1] = nn.Linear(num_ftrs, n_class) 
        elif emb_type == "efficientnet" :
            self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_class)
            num_ftrs = self.encoder._fc.in_features
            self.encoder._fc = nn.Linear(num_ftrs, n_class)
        else : # efficientnet with group normalization
            self.encoder = EfficientNet_GN.from_pretrained('efficientnet-b0', num_classes=n_class)
            num_ftrs = self.encoder._fc.in_features
            self.encoder._fc = nn.Linear(num_ftrs, n_class)


    def forward(self, img):
        x = self.encoder(img).squeeze(-1).squeeze(-1)

        if self.emb_type == 'resnet18':
            x = F.relu(x)
            x = self.linear(x)

        return x 

##############################################################################################################
