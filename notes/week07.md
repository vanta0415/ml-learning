第7周笔记

日期：2026-09-10
代码：deep_learning/training_template/

1. 这周做了啥

把前面几周的训练流程拆成工程模板：
config.py / dataset.py / model.py / train.py / test.py
入口：run_week07.py

做了：DataLoader、简单数据增强、保存 best checkpoint、训练日志、混淆矩阵。

2. 概念

(1) Dataset / DataLoader
Dataset 负责取一张样本，DataLoader 负责按 batch 取、打乱。

(2) 数据增强
训练时随机旋转一点，测试不做增强。目的是让模型见更多变化，减轻过拟合。

(3) Checkpoint
验证集最好时把权重存成 best_model.pth，测试时再加载。

(4) 混淆矩阵
行是真实类别，列是预测类别。对角线是对的，看哪些数字容易互相认错。

3. 怎么跑

cd deep_learning/training_template
python run_week07.py

或分开：
python train.py
python test.py

4. 结果

output_week07：
train_log.csv
train_loss.png
val_acc.png
best_model.pth（本地保存，gitignore）
test_result.csv
confusion_matrix.csv
confusion_matrix.png

这次数值：
best_val约0.952，test_acc约0.953（MNIST子集，4轮，有旋转增强）。

5. 一点体会

代码拆开以后，改配置、换模型、只测不训都方便很多，后面做项目也能接着用这套结构。
