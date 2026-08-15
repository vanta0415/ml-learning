第3周笔记

日期：2026-08-15
代码：deep_learning/week03_dl_basics.py

1. 这周做了啥

用PyTorch做线性回归，数据大概是 y=2x+1。
做了两组对比：
(1) 学习率 0.01 和 0.1
(2) batch size 16 和 64

2. 概念

(1) 张量
PyTorch里的数组，用来存数据和参数。

(2) 模型
这里是直线 y=wx+b。

(3) loss
预测和真实值差多少，这里用MSE。

(4) 学习率 lr
每次更新参数步子多大。太小学得慢，合适时降得更快。

(5) batch size
每次拿多少条数据更新一次。
batch小：同一轮里更新次数更多；batch大：更新更稳但次数少。

(6) 一次训练
预测 -> 算loss -> backward -> step。

3. 结果

loss_curve.png：lr=0.1 比 0.01 降得更快，w、b也更接近2和1。
loss_curve_batch.png：batch=16 和 batch=64 都能下降，曲线不完全一样。
fit_result.png：拟合直线和数据点比较接近。

4. 一点体会

学习率和batch都会影响训练快慢和曲线形状。
这周先把训练循环跑通，后面分类任务还会再用到这些参数。
