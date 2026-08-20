# 第4周：MLP做MNIST分类

from pathlib import Path
import csv
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

out_dir = Path(__file__).parent / "output_week04"
out_dir.mkdir(exist_ok=True)
data_dir = Path(__file__).parent / "data"
raw_dir = data_dir / "MNIST" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")
torch.manual_seed(0)

# 手动从镜像下载，避免官方源太慢
files = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

print("检查 MNIST...")
for name, url in files.items():
    path = raw_dir / name
    if path.exists() and path.stat().st_size > 1000:
        print("已有", name)
        continue
    print("下载", name)
    urllib.request.urlretrieve(url, path)

transform = transforms.ToTensor()
# download=True 只会在缺文件时下载；你已有raw，这里主要是解压生成processed
train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
test_full = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

# 用一部分数据，CPU上更快
train_set = Subset(train_full, list(range(8000)))
val_set = Subset(train_full, list(range(8000, 10000)))
test_set = Subset(test_full, list(range(2000)))


class MLP(nn.Module):
    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_one(name, hidden=128, dropout=0.0, lr=0.1, batch_size=128, epochs=4):
    torch.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = MLP(hidden=hidden, dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses, val_accs = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        train_losses.append(total_loss / n)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_accs.append(correct / total)
        print("%s epoch%d loss=%.4f val=%.3f" % (name, epoch + 1, train_losses[-1], val_accs[-1]))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    test_acc = correct / total
    print("%s test_acc=%.3f" % (name, test_acc))
    return {
        "name": name,
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "test_acc": test_acc,
    }


print("开始两组MLP实验")
exp_a = run_one("no_dropout", dropout=0.0)
exp_b = run_one("with_dropout", dropout=0.3)

with open(out_dir / "acc_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name", "hidden", "dropout", "lr", "batch_size", "test_acc"])
    for e in [exp_a, exp_b]:
        w.writerow([e["name"], e["hidden"], e["dropout"], e["lr"], e["batch_size"], "%.4f" % e["test_acc"]])

plt.figure()
plt.plot(exp_a["train_losses"], label="no dropout")
plt.plot(exp_b["train_losses"], label="dropout=0.3")
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.title("week04 train loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "train_loss.png", dpi=120)

plt.figure()
plt.plot(exp_a["val_accs"], label="no dropout")
plt.plot(exp_b["val_accs"], label="dropout=0.3")
plt.xlabel("epoch")
plt.ylabel("val acc")
plt.title("week04 val accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "val_acc.png", dpi=120)

print("结果在", out_dir)
print("第4周完成")
