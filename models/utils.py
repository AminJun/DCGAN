from torch import nn as nn
from models.dcgan import discriminator, generator


def parallel_cuda(func):
    def to_parallel() -> nn.Module:
        model = func()
        model = nn.DataParallel(model.cuda())
        return model

    return to_parallel


@parallel_cuda
def get_dcgan_dis_cifar10() -> nn.Module:
    return discriminator()


def get_dcgan_gen_cifar10() -> nn.Module:
    return generator()
