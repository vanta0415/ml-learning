# 第6周：ResNet迁移学习（MNIST）

from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

out_dir = Path(__file__).parent / "output_week06"
out_dir.mkdir(exist_ok=True)
data_dir = Path(__file__).parent / "data"

device = torch.device("cpu")
torch.manual_seed(0)

# ResNet要3通道，并且尺寸大一点
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

train_full = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=transform)
test_full = datasets.MNIST(root=str(data_dir), train=False, download=False, transform=transform)

train_set = Subset(train_full, list(range(4000)))
val_set = Subset(train_full, list(range(4000, 5000)))
test_set = Subset(test_full, list(range(1000)))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


def build_model(pretrained=True):
    # 新版torchvision用weights；下载失败就改成不加载预训练
    try:
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            print("已加载预训练ResNet18")
        else:
            model = models.resnet18(weights=None)
            print("使用随机初始化ResNet18")
    except Exception as e:
        print("预训练权重下载失败，改用随机初始化:", e)
        model = models.resnet18(weights=None)

    # 冻住前面特征层，只训练最后分类头
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    return model


model = build_model(pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.fc.parameters(), lr=0.001)

train_losses, val_accs = [], []
epochs = 3

print("开始迁移学习训练...")
for epoch in range(epochs):
    model.train()
    total, n = 0.0, 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    train_losses.append(total / n)

    model.eval()
    correct, total_n = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total_n += len(yb)
    val_accs.append(correct / total_n)
    print("epoch%d loss=%.4f val=%.3f" % (epoch + 1, train_losses[-1], val_accs[-1]))

model.eval()
correct, total_n = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total_n += len(yb)
test_acc = correct / total_n
print("test_acc=%.3f" % test_acc)

with open(out_dir / "acc_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "pretrained", "test_acc", "epochs", "train_size"])
    w.writerow(["ResNet18-fc", "try_yes", "%.4f" % test_acc, epochs, 4000])

plt.figure()
plt.plot(train_losses)
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.title("week06 transfer train loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "train_loss.png", dpi=120)

plt.figure()
plt.plot(val_accs)
plt.xlabel("epoch")
plt.ylabel("val acc")
plt.title("week06 transfer val acc")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "val_acc.png", dpi=120)

# 经典网络对比表（笔记里也会写）
with open(out_dir / "classic_cnn_compare.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["network", "year", "point"])
    w.writerow(["AlexNet", "2012", "深度CNN开始火起来"])
    w.writerow(["VGG", "2014", "用很多小卷积堆很深"])
    w.writerow(["GoogLeNet", "2014", "Inception多尺度特征"])
    w.writerow(["ResNet", "2015", "残差连接，更深也能训"])

print("结果在", out_dir)
print("第6周完成")
