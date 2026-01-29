

from torch import nn
import torch
class CauchyActivation_fixed(nn.Module):
    """
    originated from https://github.com/ashishbamania/XNet-vs.-ReLU-MLP
    """
    def __init__(self,l1,d):
        super().__init__()

        # Initializing λ1, λ2, d as trainable parameters
        # We initialize them for each instance of the activation
        self.lambda1 = l1
        # self.lambda2 = l2
        self.d = d

    def forward(self, x):
        # Ensure parameters are on the same device as input x
        # lambda1 = torch.pow(self.lambda1.to(x.device),1/3)
        # lambda1 = math.log(1.5)

        # if (self.lambda2 < 0):
        #     lambda2 = torch.log(-self.lambda2.to(x.device))
        # elif (self.lambda2 > 0):
        #     lambda2 = torch.log(self.lambda2.to(x.device))
        # else:
        #     lambda2 = torch.zeros(x.shape)

        # lambda2 = 0.0001 * torch.tanh(self.lambda2.to(x.device))
        lambda1 = self.lambda1
        # lambda2 = self.lambda2
        d = self.d

        x2_d2 = x ** 2 + d ** 2
        epsilon = 1e-4

        term1 = (lambda1 * x) / (x2_d2 + epsilon)
        # term2 = lambda2 / (x2_d2+ epsilon)

        return term1 #+ term2


class CNN_Relu(nn.Module):
    def __init__(self):
        super(CNN_Relu, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x

class CNN_Tanh(nn.Module):
    def __init__(self):
        super(CNN_Tanh, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.Tanh(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        x=self.conv(x)
        return x


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class MNIST_CNN_Relu(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN_Relu, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.ReLU()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MNIST_CNN_Tanh(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #return nn.functional.softmax(x)
        return x



class MNIST_CNN_Cauchy_fixed(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN_Cauchy_fixed, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.c_param_dict = {
            "features.1.lambda1": 2.0,
            "features.1.d": 0.5,
            "features.4.lambda1": 0.5,
            "features.4.d": 2.0,
            "classifier.1.lambda1": 4.0,
            "classifier.1.d": 1.0
        }
        # self.c_param_dict = {
        #     "features.1.lambda1": 2.0,
        #     "features.1.d": 0.5,
        #     "features.4.lambda1": 1.0,
        #     "features.4.d": 2.0,
        #     "classifier.1.lambda1": 5.0,
        #     "classifier.1.d": 1.0
        # }
        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            elif v[0] == 16:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)
                layers += [conv2d, CauchyActivation_fixed(self.c_param_dict.get("features.1.lambda1"),
                                                          self.c_param_dict.get("features.1.d"))]  # Replaced ReLU
                c = filters
            elif v[0] == 32:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)
                layers += [conv2d, CauchyActivation_fixed(self.c_param_dict.get("features.4.lambda1"),
                                                          self.c_param_dict.get("features.4.d"))]  # Replaced ReLU
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        CauchyActivation_fixed(self.c_param_dict.get("classifier.1.lambda1"),
                                                               self.c_param_dict.get("classifier.1.d")),
                                        # Replaced ReLU
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #return nn.functional.softmax(x)
        return x


class CIFAR10_CNN_Relu(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN_Relu, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CIFAR10_CNN_Tanh(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

