---
title: pytorch小总结
date: 2018-01-30 10:15:44
tags: ["pytorch"]
category: 深度学习框架
---
#Pytorch
这个框架确实用起来非常好用，有很多东西都封装好了。
以下是近来用到的几种方法和API。
<!-- more -->
#数据预处理
对于数据的预处理主要用到**torchvision.transforms**这里面的函数，其中一个很方便的，构造预处理拓扑的方式是调用

>class torchvision.transforms.Compose(transforms)

Parameters:	transforms (list of Transform objects) – list of transforms to compose.

一个简单的预处理例子：

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])
	train_transform = transforms.Compose(
	    [
	     transforms.RandomCrop((224,224)),
	     transforms.ToTensor(),
	     normalize
	    ]
	)
	valid_transform = transforms.Compose(
	    [
	     transforms.Resize((224,224)),
	     transforms.ToTensor(),
	     normalize
	    ]
	)

#数据读写
最常用的就是继承**torch.utils.data**里面的

>class torch.utils.data.Dataset

这个类，并且重写其中的**__len__()和__getitem__()**这个两个函数

然后使用

>class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)

Parameters:	

+ dataset (Dataset) – dataset from which to load the data.
+ batch_size (int, optional) – how many samples per batch to load (default: 1).
+ shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
+ sampler (Sampler, optional) – defines the strategy to draw samples from the dataset. If specified, shuffle must be False.
+ batch_sampler (Sampler, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
+ num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process (default: 0)
+ collate_fn (callable, optional) – merges a list of samples to form a mini-batch.
+ pin_memory (bool, optional) – If True, the data loader will copy tensors into CUDA pinned memory before returning them.
+ drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)

来封装成数据读取器用以批训练。一般可以使用**enumerate()和iter()**进行迭代循环。

阅读源码不难发现，其实**torch.utils.data.Dataset**这个类初始化工作主要就是将数据文件的路径收集起来，然后其中的**__getitem__()**根据收集的文件路径负责打开数据文件，并进行一系列的数据预处理。而**torch.utils.data.DataLoader**主要就是根据**batch_size**调用**__getitem__()**。

一个简单的自定义数据集例子：

**这个例子是当初想自己实现分离训练集和验证集的代码，但是后来发现不那么好在一个dataset中分离，所以就弃坑了。代码并不能直接用，只是提供一个重写的思路。**

	import torch
	import torch.utils.data as Data
	import torchvision.transforms as T
	import os
	import numpy as np
	from PIL import Image
	class ImageDataSet_Train(Data.Dataset):
	
	def __init__(self,root,transforms=[],valid_rate=0.1):
		transforms_error_msg = "either train_transforms & valid_transforms, or nothing"
		valid_rate_error_msg = "valid_rate should be in the range [0,1]"
		assert ((len(transforms)==0) or (len(transforms)==2)), transforms_error_msg
		assert ((valid_size >= 0) and (valid_size <= 1)), valid_rate_error_msg
		normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
		# load train_data and then split it into train_set and valid_set according to valid_rate

		label_folders = [os.path.join(root,img) for img in os.listdir(root)]
		self.labels = [label.split('/')[-1] for label in label_folders]
		imgs = [os.path.join(l,f) for l in label_folders for f in os.listdir(l)]
		imgs = np.random.permutation(imgs)
		set_size = len(imgs)
		#calculate the size of valid_set and store the start and end index of valid_set[start,end)
		split = np.floor(valid_rate*set_size)
		self.valid_start = 0
		self.valid_end = split
		self.valid_set = imgs[self.valid_start:self.valid_end]
		
		#concatenate the rest data as train_set
		self.train_set = imgs[0:self.valid_start].extend(imgs[self.valid_end:])
		
		#if no transforms is given
		if len(transforms) == 0:
			self.train_transforms = T.Compose([
				T.Resize([256,256]),
				T.RandomCrop(32,padding=4),
				T.RandomHorizontalFlip(),
				T.ToTensor(),
				normalize
				])
			self.valid_transforms = T.Compose([
				T.Resize([256,256]),
				T.ToTensor(),
				normalize
				])
		else:
			self.train_transforms = transforms[0]
			self.valid_transforms = transforms[1]
		
	def __getitem__(self,index):
		img_path = self.train_set[index]
		'''关键的一步，根据路径和命名规则分离出标签，并转换相应的编号'''
		label = self.labels.index(img_path.split('.'))
		data = Image.open(img_path)
		data = self.train_transforms(data)
		return data,label


对于图片数据集，pytorch有一个现成的API可以使用，它会自动读取形如：/train/label/ 的数据集，然后自动对各个label进行编号（从0开始）。

>class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)

Parameters:	

+ root (string) – Root directory path.
+ transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
+ target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
+ loader – A function to load an image given its path.

一个简单的使用读取并分离出训练集和验证集的例子：

	data_dir = #where your data dir locates
	'''训练集需要做增强处理，验证集不需要，但对两者原始样本的改动必须保持一直'''
	train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transform)
	valid_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=valid_transform)
	
	'''分类数目，标签可以直接查看classes这个属性'''
	nclasses = len(train_dataset.classes)
	num_train = len(train_dataset)#样本数

	'''构造训练集乱序索引，并根据指定的验证集比例大小，划分一部分索引子集作为验证集索引'''
	indices = list(range(num_train))
	split = int(np.floor(VALID_SIZE * num_train))
	np.random.seed(0)
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]

	'''这里用到了SubsetRandomSampler来根据传入的索引返回子集'''
	train_sampler = Data.sampler.SubsetRandomSampler(train_idx)
	valid_sampler = Data.sampler.SubsetRandomSampler(valid_idx)

	'''构造迭代器'''
	train_loader = Data.DataLoader(train_dataset, 
	                    batch_size=BATCH_SIZE,sampler=train_sampler,
	                    num_workers=2, pin_memory=True,)
	valid_loader = Data.DataLoader(valid_dataset, 
	                    batch_size=BATCH_SIZE, sampler=valid_sampler, 
	                    num_workers=2, pin_memory=True)

#调用实现好的模型
ResNet50

>torchvision.models.resnet50(pretrained=False, \*\*kwargs)

一个默认参数pretrained表示是否使用预训练模型，也就是从指定URL下载已经训练好的模型参数。

对应源码如下：

	__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
	           'resnet152']


	model_urls = {
	    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
	}
---
	def resnet50(pretrained=False, **kwargs):
	    """Constructs a ResNet-50 model.

	    Args:
	        pretrained (bool): If True, returns a model pre-trained on ImageNet
	    """
	    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	    if pretrained:
	        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	    return model

另一个则是关键字参数，在**model = ResNet(Bottleneck, [3, 4, 6, 3], \*\*kwargs)**这里会被传入。

注意到

	class ResNet(nn.Module):

	    def __init__(self, block, layers, num_classes=1000):
	        self.inplanes = 64
	        super(ResNet, self).__init__()
	        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
	                               bias=False)
	        self.bn1 = nn.BatchNorm2d(64)
	        self.relu = nn.ReLU(inplace=True)
	        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
	        self.layer1 = self._make_layer(block, 64, layers[0])
	        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
	        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
	        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
	        self.avgpool = nn.AvgPool2d(7, stride=1)
	        self.fc = nn.Linear(512 * block.expansion, num_classes)

	        for m in self.modules():
	            if isinstance(m, nn.Conv2d):
	                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
	                m.weight.data.normal_(0, math.sqrt(2. / n))
	            elif isinstance(m, nn.BatchNorm2d):
	                m.weight.data.fill_(1)
	                m.bias.data.zero_()

**__init__()**函数中存在num_classes这个默认参数，因此我们可以通过\*\*kwargs这个关键字参数传递待分类的标签数目。比如我们希望num_classes=10，那么如此调用即可。

	net = models.resnet50(num_classes=10)

#分类任务中计算Accuracy
在分类中一般使用**torch.max()**将网络的输出结果（标签向量）中预测概率最大的索引提取出来，并压榨(squeeze)成一维向量，然后统计正确率。
另外需要注意的是，如果网络模型中包含dropout和batchnorm等只在训练中启用的层，那么需要使用net.eval()将网络模型转为测试模式以禁用他们，测试后再调用net.train()启用他们继续训练。

一个简单的例子：

	if step % 5 == 4:
	 net.eval()
	 correct = 0
	 for _,(t_x,t_y) in enumerate(valid_loader):
	     test_x = Variable(t_x).cuda()
	     test_y = Variable(t_y).cuda()
	     test_output = net(test_x)
	     pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
	     correct += torch.sum(pred_y == test_y.data)
	 print("Epoch: %d Step: %d Loss: %f Accuracy: %f" % (epoch + 1,step + 1,running_loss/5,correct/split))
	 running_loss = 0
	 net.train()

#使用指定GPU
使用编号为6，7的显卡：

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
