import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
import torch
import torchvision
from torchvision import datasets, transforms
from bnn import BConfig, prepare_binary_model
# Import a few examples of quantizers
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)

# Create your desire model (note the default R18 may be suboptimal)
# additional binarization friendly models are available in bnn.models


# Define the binarization configuration and assign it to the model
bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    # optionally, one can pass certain custom variables
    weight_pre_process=XNORWeightBinarizer.with_args(center_weights=True)
)
# Convert the model appropiately, propagating the changes from parent node to leafs
# The custom_config_layers_name syntax will perform a match based on the layer name,
# setting a custom quantization function.

if __name__ == "__main__":

    # 超参数定义
    EPOCH = 50
    BATCH_SIZE = 128
    LR = 0.001
    # 数据集加载
    # 对训练集及测试集数据的不同处理组合

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转 选择一个概率概率
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
    train_data = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=True)
    test_data = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_test, download=True)

    # 数据分批


    # 使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
    # shuffle表示在每个epoch开始的时候，对数据进行重新排序
    # 数据分批之前：torch.Size([3, 32, 32])：Tensor[[32*32][32*32][32*32]],每一个元素都是归一化之后的RGB的值；数据分批之后：torch.Size([64, 3, 32, 32])
    # 数据分批之前：train_data([50000[3*[32*32]]])
    # 数据分批之后：train_loader([50000/64*[64*[3*[32*32]]]])
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 模型加载，有多种内置模型可供选择
    # model = torchvision.models.densenet201(pretrained=False)
    model = models.resnet18(weights=None)
    # 修改模型
    model.conv1 = nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=1, bias=False)  # 首层改成3x3卷积核
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
    num_ftrs = model.fc.in_features  # 获取（fc）层的输入的特征数
    model.fc = nn.Linear(num_ftrs, 10)
    model = prepare_binary_model(model, bconfig, custom_config_layers_name=[{'conv1': BConfig()}])


    # 定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
    criterion = nn.CrossEntropyLoss()
    # torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 设置GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    # 模型和输入数据都需要to device
    model = model.to(device)

    # 模型训练
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('cifar-10')
    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            # 取出数据及标签
            inputs, labels = data
            # 数据及标签均送入GPU或CPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            # 计算损失函数
            loss = criterion(outputs, labels)
            # 清空上一轮的梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 利用tensorboard，将训练数据可视化
            if i % 50 == 0:
                writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)
            # print('it’s training...{}'.format(i))

        sum_loss = 0
        accurate = 0
        with torch.no_grad():
            for data in test_loader:
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
                imgs, targets = data
                imgs = imgs.cuda()
                targets = targets.cuda()
                output = model(imgs)
                loss_in = criterion(output, targets)

                sum_loss += loss_in
                # print('Output:', output)
                accurate += (output.argmax(1) == targets).sum()

        print('epoch{} accuracy:{:.2f}%'.format(epoch + 1, accurate / len(test_data) * 100))
        print('epoch{} loss:{:.4f}'.format(epoch + 1, loss.item()))

        writer.add_scalar('Valid/Loss', sum_loss, epoch)
        writer.add_scalar('Valid/Accuracy', accurate / len(test_data) * 100, epoch)






    # You can also ignore certain layers using the ignore_layers_name.
    # To pass regex expression, frame them between $ symbols, i.e.: $expression$.
    print("end")


