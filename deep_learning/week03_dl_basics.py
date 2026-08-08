# 第3周：深度学习训练流程（线性回归）
# 运行: python week03_dl_basics.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

out_dir = Path(__file__).parent / "output_week03"
out_dir.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# 1) 造一批假数据：y = 2x + 1 + 一点噪声
x = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
y = (2 * x + 1 + 0.1 * np.random.randn(*x.shape)).astype(np.float32)

x_t = torch.from_numpy(x)
y_t = torch.from_numpy(y)


def train_once(lr, epochs=80):
    # 最简单的模型：y = wx + b
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

    # 看学到的 w, b 接近不接近 2 和 1
    w = model.weight.item()
    b = model.bias.item()
    return losses, w, b


print("开始训练...")
loss_small, w1, b1 = train_once(lr=0.01)
loss_big, w2, b2 = train_once(lr=0.1)

print("学习率0.01 -> w=%.3f, b=%.3f, 最后loss=%.4f" % (w1, b1, loss_small[-1]))
print("学习率0.1  -> w=%.3f, b=%.3f, 最后loss=%.4f" % (w2, b2, loss_big[-1]))

# 2) 画 loss 曲线
plt.figure()
plt.plot(loss_small, label="lr=0.01")
plt.plot(loss_big, label="lr=0.1")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("week03 loss curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=120)
print("已保存:", out_dir / "loss_curve.png")

# 3) 再画一下拟合效果（用 lr=0.1 那个）
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
plt.scatter(x, y, s=12, label="data")
plt.plot(x, y_pred, color="red", label="fit")
plt.legend()
plt.title("linear regression fit")
plt.tight_layout()
plt.savefig(out_dir / "fit_result.png", dpi=120)
print("已保存:", out_dir / "fit_result.png")
print("第3周完成")
