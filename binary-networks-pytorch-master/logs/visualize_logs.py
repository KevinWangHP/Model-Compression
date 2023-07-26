import math

from matplotlib import rcParams
import matplotlib.pyplot as plt
import re
import os

##显示中文
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = 'SimSun,Times New Roman'

def load_acc_loss(filename):
    os.makedirs(filename, exist_ok=True)

    ##读取log文件
    logFile = filename + ".log"
    all_list = []
    file = open(logFile)
    for line in file:
        all_list.append(line)
    file.close()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    variance = []
    best_acc = 0
    for i in all_list:
        if "Train acc" in i:
            train_acc.append(float(i.split('Train acc: ')[1].split(', train loss')[0]))
            train_loss.append(float(i.split('train loss:')[1]))
        if "Current acc:" in i:
            val_acc.append(float(i.split('Current acc: ')[1].split(', current loss:')[0]))
            val_loss.append(float(i.split('current loss:')[1].split(', best acc:')[0]))
            best_acc = float(i.split('best acc: ')[1])
        if "Gradient Variance" in i:
            epoch = int(i.split("Epoch")[1].split(" Gradient")[0])
            var = float(i.split(": ")[1])
            variance.append([epoch, var])
    return train_acc, val_acc, train_loss, val_loss, variance, best_acc

def draw(filename, content):
    res = load_acc_loss(filename)
    if content == "acc":
        plt.title("Accuracy")
        plt.plot(res[0], label="Train_"+filename)
        plt.plot(res[1], label="Test_"+filename)
    elif content == "loss":
        plt.title("Loss")
        plt.plot(res[2], label="Train_"+filename)
        plt.plot(res[3], label="Test_"+filename)
    elif content == "variance":
        plt.title("Variance")
        plt.plot([i[0] for i in res[4]], [math.log10(i[1]) for i in res[4]], label=filename)
    elif content == "best":
        plt.title("Best Accuracy")
        plt.xlabel('Accuracy', fontsize=14)
        plt.xlabel('Bit', fontsize=14)
        plt.xlim(-1, 10)
        plt.ylim(0, 100)
        plt.scatter(int(filename.split("_")[-1].split("bit")[0]), res[5], s=200)

def draw_curve(curve_name):
    draw("PSQ_visualize_8bit", curve_name)
    draw("PSQ_visualize_4bit", curve_name)
    draw("PSQ_visualize_2bit", curve_name)
    draw("PSQ_visualize_1bit", curve_name)

draw_curve("variance")
plt.legend(loc="best")
plt.show()



