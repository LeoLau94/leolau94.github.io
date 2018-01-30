---
title: pytorch中nn&nn.functional的设计机制
date: 2018-01-30 15:22:25
tags: ["pytorch"]
category: 深度学习框架
---
之前对torch.nn中已经封装成类的层，却在nn.functional中又提供以函数的形式进行调用的接口，感到十分不解。后来看了源码后才明白Pytorch如此设计的精妙之处。
<!--more-->

#使用的设计机制
以官方文档中的教程为例子：
```
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```
注意到其中的语句**`x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))`**和**x = F.relu(self.fc1(x))**实现的是MaxPool2d层和ReLU层的功能。

查看**nn.MaxPool2d**源码
```
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
```
可以发现，其实**nn.MaxPool2d**层的**forward**函数就是调用的**F.max_pool2d**。类似的可以发现，其他种类的层，也是采取这样的机制。

#那么为什么要这样设计呢？
根据网上搜索到的说法，主要的原因就是为了兼顾网络模型定义时的**灵活性**和**便利性**

>在建图过程中，往往有两种层，一种如全连接层，卷积层等，当中有Variable，另一种如Pooling层，Relu层等，当中没有Variable。
如果所有的层都用nn.functional来定义，那么所有的Variable，如weights，bias等，都需要用户来手动定义，非常不方便。
而如果所有的层都换成nn来定义，那么即便是简单的计算都需要建类来做，而这些可以用更为简单的函数来代替的。
所以在定义网络的时候，如果层内有Variable,那么用nn定义，反之，则用nn.functional定义。

>转自：[CSDN](http://blog.csdn.net/GZHermit/article/details/78730856)
