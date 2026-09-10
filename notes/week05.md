第5周笔记

日期：2026-08-30
代码：deep_learning/week05_cnn_baseline.py

1. 这周做了啥

用一个小CNN在MNIST上做分类。
保存了训练曲线、验证准确率，还有预测对/错的样例图。

2. 概念

(1) 卷积
用小卷积核在图上滑，提取局部特征。

(2) 池化
缩小特征图，常见是MaxPool。

(3) CNN和MLP的区别
MLP先把图拉平，容易丢掉空间结构。
CNN保留平面结构，更适合图像。

3. 结果

output_week05：
acc_table.csv
train_loss.png
val_acc.png
pred_correct.png
pred_wrong.png

测试准确率约 0.950（见acc_table.csv）。
错分样例在pred_wrong.png，能看出哪些数字容易混。

4. 一点体会

看错分样例能知道模型容易把哪些数字搞混。
