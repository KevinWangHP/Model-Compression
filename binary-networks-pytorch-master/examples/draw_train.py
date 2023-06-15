from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
import os


def read_file(file_name, flag):
    ##读取log文件
    logFile = file_name
    text = ''
    file = open(logFile)
    train_line = []
    test_line = []
    for line in file:
        if "Train acc:" in line:
            train_line.append(line)
        elif "Current acc:" in line:
            test_line.append(line)
    file.close()

    # all_list = re.findall('Current acc:', text)

    train_loss = []
    for i in train_line:
        train_loss.append(float(i.split('Train loss: ')[1].split(', Epoch time:')[0]))

    train_acc = []
    for i in train_line:
        train_acc.append(float(i.split('Train acc: ')[1].split(', Train loss:')[0]))

    val_loss = []
    for i in test_line:
        if flag == 0:
            val_loss.append(float(i.split('Current loss: ')[1].split(', Best acc:')[0]))
        else:
            val_loss.append(float(i.split('Current loss: ')[1].split(', best acc:')[0]))

    val_acc = []
    for i in test_line:
        val_acc.append(float(i.split('Current acc: ')[1].split(', Current loss:')[0]))

    return train_loss, train_acc, val_loss, val_acc


if __name__ == '__main__':


    ##显示中文
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'SimSun,Times New Roman'

    resnet18_train_loss, resnet18_train_acc, resnet18_test_loss, resnet18_test_acc =\
        read_file(r'result\nohup_resnet18.txt', 0)
    xornet_train_loss, xornet_train_acc, xornet_test_loss, xornet_test_acc =\
        read_file(r'result\nohup_xornet.txt', 1)

    os.makedirs("draft", exist_ok=True)

    epoch = [i for i in range(50)]

    blocks_list = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    plt.plot(epoch, xornet_test_acc, lw=1, c='blue', marker='s', ms=2, label="XNORNET")
    plt.plot(epoch, resnet18_test_acc, lw=1, c='orange', marker='o', ms=2, label="RESNET18")
    # plt.plot(xlist, supervised_res_vit, lw=1, c='orange', marker='s', ms=5, label="VIT_supervised")
    # plt.plot(xlist, unsupervised_res_vit, lw=1, c='orange', marker='o', ms=5, label="VIT_unsupervised", linestyle='--')
    # plt.plot(xlist, supervised_res_WRN50[:len(xlist)], lw=1, c='red', marker='s', ms=5, label="WRN50_supervised")
    # plt.plot(xlist, unsupervised_res_WRN50[:len(xlist)], lw=1, c='red', marker='o', ms=5, label="WRN50_unsupervised", linestyle='--')
    # plt.plot(blocks_list, average_res, lw=1, c='red', marker='o', ms=10, label="average")
    # plt.tick_params("x", labelsize=12)
    # plt.tick_params("y", labelsize=20)
    # plt.plot(blocks_list, average_res, label="average")
    # plt.axhline(supervised_res[0], lw=1, linestyle='--', c="red", label="unsupervised")
    # plt.axhline(unsupervised_res[0], lw=1, linestyle='--', c="red", label="unsupervised")
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel("Acuracy", fontsize=15, rotation=0)
    plt.title("ACCURACY", fontsize=40)
    plt.legend(prop={'size': 10})
    plt.savefig("draft/ACCURACY")
    plt.show()


