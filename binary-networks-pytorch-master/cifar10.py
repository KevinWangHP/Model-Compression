import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import time
import numpy as np
import random
from bnn.layers.debug import dump, plot_bin_hist, fast_dump_2



from bnn.ops import  (
    BasicInputBinarizer,
    XNORWeightBinarizer
)
from bnn import BConfig, prepare_binary_model, Identity
from  bnn.models.resnet import resnet18

from examples.utils import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--print_freq', type=int, default=100,
                    help='logs printing frequency')
parser.add_argument('--out_dir', type=str, default='')
parser.add_argument('--optimizer', default= 'adam', type = str)
parser.add_argument('--pretrained', default= True, type=bool)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='/data/datasets/cifar10/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='/data/datasets/cifar10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = resnet18()
# net = models.resnet18(pretrained=True)
net.conv1 = nn.Conv2d(net.conv1.in_channels, net.conv1.out_channels, (3, 3), (1, 1), 1)
net.maxpool = nn.Identity()  # nn.Conv2d(64, 64, 1, 1, 1)
net.fc = nn.Linear(net.fc.in_features, 10)
#net = torch.nn.DataParallel(net)
# if args.pretrained == True:
#     checkpoint = torch.load('./checkpoint/ckpt_full.pth')
#     checkpoint = checkpoint['net']
#     net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
#checkpoint = torch.load('./checkpoint/ckpt_full.pth')
#net.load_state_dict(checkpoint['net'])
#net = net.module()
# Binarize
print('==> Preparing the model for binarization')
bconfig = BConfig(
            activation_pre_process = BasicInputBinarizer,
            activation_post_process = Identity,
            weight_pre_process = XNORWeightBinarizer
        )
# first and last layer will be kept FP32
model = prepare_binary_model(net, bconfig, custom_config_layers_name={'conv1': BConfig(), 'fc': BConfig()})
print(model)

net = net.to(device)
if 'cuda' in device:
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
else:
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 140, 180], 0.1)

# Training
def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    print('\nTrain Epoch: %d' % epoch)
    net.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, = accuracy(outputs, targets)

        top1.update(acc1.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)
    acc = top1.avg
    train_loss = losses.avg
    print('Train acc: {}, train loss:{}'.format(acc, train_loss))
    torch.cuda.empty_cache()


def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, data_time, losses, top1],
        prefix="Test Epoch: [{}]".format(epoch))

    global best_acc
    net.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets)

            top1.update(acc1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    # Save checkpoint.
    acc = top1.avg
    test_loss = losses.avg
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    print('Current acc: {}, current loss:{}, best acc: {}'.format(acc, test_loss, best_acc))


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




seed_torch()
for epoch in range(start_epoch, start_epoch+200):
    if epoch % 5 == 0 or epoch < 10:
        plot_bin_hist(net, optimizer, testloader, "visualize/",
                      device, criterion, epoch)
    train(epoch)
    scheduler.step()
    test(epoch)


# print(args.pretrained, args.optimizer)
