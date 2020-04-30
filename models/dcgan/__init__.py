from torch import nn as nn
from torch.utils import model_zoo

from models.dcgan.generator import Generator
from models.dcgan.discriminator import Discriminator

_G_PATH = 'https://github.com/AminJun/DCGAN/releases/download/DCGAN1/netG_epoch_199.pth'
_D_PATH = 'https://github.com/AminJun/DCGAN/releases/download/DCGAN1/netD_epoch_199.pth'


def weights_init(model: nn.Module) -> nn.Module:
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    return model


def normal_weight(func_model):
    def make_normal() -> nn.Module:
        model = func_model()
        return weights_init(model)

    return make_normal


@normal_weight
def generator(pretrained: bool = True) -> nn.Module:
    model = Generator()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(_G_PATH))
    return model


@normal_weight
def discriminator(pretrained: bool = True) -> nn.Module:
    model = Discriminator()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(_D_PATH))
    return model
