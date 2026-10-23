"""
Model definitions used by FedLoRD. All four are copied verbatim from PFLlib
(https://github.com/TsingZ0/PFLlib, Apache-2.0):

  FedAvgCNN   -- the two-conv CNN of McMahan et al. (2017)   (-m CNN;  CIFAR-10/100, Omniglot, MNIST)
  resnet18_gn -- torchvision ResNet-18 with GroupNorm and a CIFAR 3x3 stem (-m ResNet18; CIFAR-100, Tiny-ImageNet)
  fastText    -- bag-of-embeddings text classifier (-m fastText; AG News)
  HARCNN      -- CNN for human-activity recognition (-m HARCNN; HAR, PAMAP2)
"""
import torch
import torch.nn.functional as F
from torch import nn


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


# ====================================================================================================================

def _convert_bn_to_gn(module, num_groups=32):
    """Recursively replace every BatchNorm2d with GroupNorm.

    GroupNorm is the standard normalization for federated learning on ResNets:
    BatchNorm's running statistics are unreliable under non-IID client data,
    whereas GroupNorm is batch-independent. num_groups is reduced to the largest
    power-of-two divisor of the channel count when 32 does not divide it.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            g = num_groups
            while C % g != 0:
                g //= 2
            setattr(module, name, nn.GroupNorm(g, C))
        else:
            _convert_bn_to_gn(child, num_groups)


def resnet18_gn(num_classes=100, small_input=True):
    """ResNet-18 with GroupNorm (the standard FL backbone for CIFAR-100).

    When small_input is True the ImageNet stem (7x7 stride-2 conv + maxpool) is
    replaced by a 3x3 stride-1 conv with no maxpool, the de-facto "CIFAR ResNet"
    stem that preserves spatial resolution for 32x32 images.
    """
    import torchvision
    m = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    if small_input:
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    _convert_bn_to_gn(m)
    return m


# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = F.log_softmax(z, dim=1)

        return out


# ====================================================================================================================

# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py
class HARCNN(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=64*26, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
