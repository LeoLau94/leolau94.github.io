---
title: 回归PyTorch
date: 2018-03-06 20:19:00
tags: ["pytorch"]
category: 深度学习框架
---
思前想后，最后还是回归到Pytorch这个框架。不得不说，比起**MXNet**，**PyTorch**的确是要灵活不少。
以前是自己半桶水，没有很好地领会到它的精髓之处。当然，现在也没有，`d=====(￣▽￣*)b`

# 一些模板代码(dirty)
<!--more-->
## 命令行参数设置
```
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from time import time

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--fine-tune', default='', type=str, metavar='PATH',
                    help='fine-tune from pruned model')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu-devices',type=str,default='0',help='decide which gpu devices to use.For exmaple:0,1')
parser.add_argument('--root',type=str,default='./', metavar='PATH', help='path to save checkpoint')
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
     torch.cuda.manual_seed(args.seed)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
     print('Using gpu devices:{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

if args.root:
	if not os.path.exists(args.root):
		os.mkdir(args.root)

```

---

2018/3/10更新

---

发现一个不知道是PyTorch(0.30)还是CUDA(8.0)引起的一个bug，就是直接再代码中设置**os.environ['CUDA_VISIBLE_DEVICES']**，有时候会失效，也就是说，无论你设置为哪块GPU，它都只使用GPU0。暂时没发现引起这个bug的原因和出现的条件。因此，在命令行指定是最保险的做法。例如：**`CUDA_VISIBLE_DEVICES=1,2 python xx.py`**

## 数据集读取(cifar10)
```
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
     datasets.CIFAR10('../data',train=True,download=False,
          transform=transforms.Compose([
               transforms.Pad(4),
               transforms.RandomCrop(32),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((.5,.5,.5),(.5,.5,.5))
               ])
          ),batch_size=args.batch_size,shuffle=True,**kwargs
     )
test_loader = torch.utils.data.DataLoader(
     datasets.CIFAR10('../data',train=False,
          transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((.5,.5,.5),(.5,.5,.5))
               ])
          ),
     batch_size = args.test_batch_size,shuffle=True,**kwargs
     )
```
##训练及测试过程
```
def train(e):
	model.train()
	correct = 0
	train_size =0
	for batch_idx,(data,label) in enumerate(train_loader):
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data),Variable(label)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,label)
		loss.backward()
		optimizer.step()
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
		train_size += len(data)
		if (batch_idx + 1) % args.log_interval == 0:
			print("Epoch: {} [{}/{} ({:.2f}%)]\t Loss: {:.6f}\t Acc: {:.6f}".format(
				e,
				(batch_idx + 1) * len(data),
				len(train_loader.dataset),
				100. * (batch_idx + 1) / len(train_loader),
				loss.data[0],
				correct / train_size
				))
			correct = 0
			train_size = 0

def test():
	model.eval()
	test_loss = 0
	correct = 0
	flag = False
	criterion.size_average=False
	start_time = time()
	for data,label in test_loader:
		if args.cuda:
			data,label = data.cuda(),label.cuda()
		data,label = Variable(data,volatile=True),Variable(label)
		output = model(data)
		test_loss += criterion(output,label).data[0]
		pred = output.data.max(1,keepdim=True)[1]
		correct += pred.eq(label.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\n Test_average_loss: {:.4f}\t Acc: {}/{} ({:.1f}%)\t Time: {:.4f}s\n'.format(
		test_loss,
		correct,
		len(test_loader.dataset),
		100. * correct / len(test_loader.dataset),
		time() - start_time,
		))
	criterion.size_average=True
	return correct / float(len(test_loader.dataset))

def save_checkpoint(state,is_best):
	file = os.path.join(args.root,'checkpoint.pkl')
	torch.save(state,file)
	if is_best:
		shutil.copyfile(file,os.path.join(args.root,'model_best.pkl'))


print(model)
print('\n-----Start Training-----\n')
for e in range(args.start_epoch,args.epochs):
	if e in [args.epochs*0.5, args.epochs*0.75]:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1
	train(e)
	precision = test()
	is_best = precision > best_precision
	training_state={
	'cfg': cfg,
	'start_epoch': e + 1,
	'model_state_dict': model.state_dict(),
	'optimizer': optimizer.state_dict(),
	'precision': precision,
	}
	save_checkpoint(
		training_state,
		is_best
		)
print("\n-----Training Completed-----\n")
```
##最近的复现剪枝的一些代码(部分)
```
print('\nPruning Start\n')
total = 0
for m in model.modules():
	if isinstance(m,nn.BatchNorm2d):
		total += m.weight.data.shape[0]
bn = torch.zeros(total)
idx = 0
for m in model.modules():
	if isinstance(m,nn.BatchNorm2d):
		size = m.weight.data.shape[0]
		bn[idx:(idx+size)] = m.weight.data.abs().clone()
		idx += size
bn_sorted,bn_sorted_idx = torch.sort(bn)
threshold_idx = int(total * args.prune_rate)
threshold = bn_sorted[threshold_idx]
print("Pruning Threshold: {}".format(threshold))

pruned = 0
cfg = []
cfg_mask = []

for i,m in enumerate(model.modules()):
	if isinstance(m,nn.BatchNorm2d):
		weight_copy = m.weight.data.clone()
		mask = weight_copy.abs().gt(threshold).float().cuda()
		pruned += mask.shape[0] - torch.sum(mask)
		m.weight.data.mul_(mask)
		m.bias.data.mul_(mask)
		cfg.append(int(torch.sum(mask)))
		cfg_mask.append(mask.clone())
		print('Layer_idx: {:d} \t Total_channels: {:d} \t Remained_channels: {:d}'.format(
			i,mask.shape[0],int(torch.sum(mask))
			))
	elif isinstance(m,nn.MaxPool2d):
		cfg.append('M')

pruned_ratio = pruned / total

print("Pre-processing done! {}".format(pruned_ratio))
```

```
layer_idx = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_idx]
change_first_linear = False

for ((i,m),m_new) in zip(enumerate(model.modules()),new_model.modules()):
	idx0 = torch.squeeze(torch.nonzero(start_mask))
	idx1 = torch.squeeze(torch.nonzero(end_mask))
	if isinstance(m,nn.BatchNorm2d):
		m_new.weight.data = m.weight.data[idx1].clone()
		m_new.bias.data = m.bias.data[idx1].clone()
		m_new.running_mean = m.running_mean[idx1].clone()
		m_new.running_var = m.running_var[idx1].clone()
		layer_idx += 1
		start_mask = end_mask.clone()
		if layer_idx < len(cfg_mask):
			end_mask = cfg_mask[layer_idx]
	elif isinstance(m,nn.Conv2d):
		w = m.weight.data[:,idx0.tolist(),:,:].clone()
		m_new.weight.data = w[idx1.tolist(),:,:,:].clone()
	elif isinstance(m,nn.Linear):
		if change_first_linear is False:
			m_new.weight.data = m.weight.data[:,idx0.tolist()].clone()
			change_first_linear = True
		else:
			pass

print('Pruning done! Channel pruning result:{}'.format(cfg))
torch.save({'cfg': cfg, 'model_state_dict':new_model.state_dict()},os.path.join(args.save,'model_pruned.pkl'))
```

##一些有用和有趣的api（持续更新）
来自[**PyTorch Document**](http://pytorch.org/docs/master/index.html)

1. `torch.masked_select(input, mask, out=None) → Tensor`

Example

```
>>> x = torch.randn(3, 4)
>>> x

 1.2045  2.4084  0.4001  1.1372
 0.5596  1.5677  0.6219 -0.7954
 1.3635 -1.2313 -0.5414 -1.8478
[torch.FloatTensor of size 3x4]

>>> mask = x.ge(0.5)
>>> mask

 1  1  0  1
 1  1  1  0
 1  0  0  0
[torch.ByteTensor of size 3x4]

>>> torch.masked_select(x, mask)

 1.2045
 2.4084
 1.1372
 0.5596
 1.5677
 0.6219
 1.3635
[torch.FloatTensor of size 7]
```