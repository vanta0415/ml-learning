第6周笔记

日期：2026-08-30
代码：deep_learning/week06_transfer_learning.py

1. 这周做了啥

(1) 整理了AlexNet/VGG/GoogLeNet/ResNet的简单对比
(2) 用ResNet18做迁移学习：冻住前面层，只训练最后的分类头
数据还是MNIST（灰度转成3通道，并resize）

2. 概念

(1) 经典网络
AlexNet：较早做出名的深度CNN
VGG：堆很多3x3卷积
GoogLeNet：Inception，多尺度
ResNet：残差连接，网络可以更深

(2) BatchNorm
让中间特征更稳定，训练更顺一点。

(3) 残差连接
加一条“抄近路”的连接，缓解深层难训的问题。

(4) 迁移学习
用别人在大数据上训好的模型，改最后一层适配自己的任务。
常会冻住前面，只训分类头，或者再微调一部分。

3. 结果

output_week06：
acc_table.csv
train_loss.png
val_acc.png
classic_cnn_compare.csv

这次数值大概是：
test_acc约0.799（小数据+只训3轮+只训练最后一层）。
预训练权重这次下载成功了。
经典网络对比表在classic_cnn_compare.csv。

4. 一点体会

同样是分类，迁移学习不用从零把所有层都训起来，省时间，也是工程里常用做法。
