# 第3周：线性回归（对比学习率和batch size）

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

out_dir = Path(__file__).parent / "output_week03"
out_dir.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# 假数据：y ≈ 2x + 1
x = np.linspace(-1, 1, 200).reshape(-1, 1).astype(np.float32)
y = (2 * x + 1 + 0.1 * np.random.randn(*x.shape)).astype(np.float32)
x_t = torch.from_numpy(x)
y_t = torch.from_numpy(y)
dataset = TensorDataset(x_t, y_t)


def train_by_epoch(lr, epochs=80):
    # 每一轮用全部数据更新一次（方便只看学习率）
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model.weight.item(), model.bias.item()


def train_by_batch(lr, batch_size, epochs=40):
    # 按 batch 训练，用来对比不同 batch size
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        losses.append(total / n)
    return losses, model.weight.item(), model.bias.item()


print("1) 对比学习率")
loss_lr1, w1, b1 = train_by_epoch(lr=0.01)
loss_lr2, w2, b2 = train_by_epoch(lr=0.1)
print("lr=0.01 -> w=%.3f b=%.3f loss=%.4f" % (w1, b1, loss_lr1[-1]))
print("lr=0.1  -> w=%.3f b=%.3f loss=%.4f" % (w2, b2, loss_lr2[-1]))

plt.figure()
plt.plot(loss_lr1, label="lr=0.01")
plt.plot(loss_lr2, label="lr=0.1")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss vs learning rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=120)
print("保存", out_dir / "loss_curve.png")

print("2) 对比 batch size")
loss_bs16, w3, b3 = train_by_batch(lr=0.05, batch_size=16)
loss_bs64, w4, b4 = train_by_batch(lr=0.05, batch_size=64)
print("bs=16 -> w=%.3f b=%.3f loss=%.4f" % (w3, b3, loss_bs16[-1]))
print("bs=64 -> w=%.3f b=%.3f loss=%.4f" % (w4, b4, loss_bs64[-1]))

plt.figure()
plt.plot(loss_bs16, label="batch=16")
plt.plot(loss_bs64, label="batch=64")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss vs batch size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "loss_curve_batch.png", dpi=120)
print("保存", out_dir / "loss_curve_batch.png")

# 拟合图（用效果较好的 lr=0.1）
torch.manual_seed(0)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(80):
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    y_pred = model(x_t).numpy()

plt.figure()
plt.scatter(x, y, s=10, label="data")
plt.plot(x, y_pred, color="red", label="fit")
plt.legend()
plt.title("linear regression fit")
plt.tight_layout()
plt.savefig(out_dir / "fit_result.png", dpi=120)
print("保存", out_dir / "fit_result.png")
print("第3周完成")
