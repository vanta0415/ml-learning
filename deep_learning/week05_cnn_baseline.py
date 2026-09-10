# 第5周：简单CNN做MNIST分类

from pathlib import Path
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

out_dir = Path(__file__).parent / "output_week05"
out_dir.mkdir(exist_ok=True)
data_dir = Path(__file__).parent / "data"

device = torch.device("cpu")
torch.manual_seed(0)

transform = transforms.ToTensor()
train_full = datasets.MNIST(root=str(data_dir), train=True, download=False, transform=transform)
test_full = datasets.MNIST(root=str(data_dir), train=False, download=False, transform=transform)

train_set = Subset(train_full, list(range(8000)))
val_set = Subset(train_full, list(range(8000, 10000)))
test_set = Subset(test_full, list(range(2000)))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SmallCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_accs = [], []
epochs = 4

print("开始训练CNN...")
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

# 测试
model.eval()
correct, total_n = 0, 0
all_x, all_y, all_p = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total_n += len(yb)
        all_x.append(xb)
        all_y.append(yb)
        all_p.append(pred)
test_acc = correct / total_n
print("test_acc=%.3f" % test_acc)

xs = torch.cat(all_x)
ys = torch.cat(all_y)
ps = torch.cat(all_p)
ok_idx = (ps == ys).nonzero(as_tuple=True)[0][:8]
bad_idx = (ps != ys).nonzero(as_tuple=True)[0][:8]


def save_grid(indices, path, title):
    n = len(indices)
    if n == 0:
        print("没有样例可保存:", path.name)
        return
    plt.figure(figsize=(8, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, 4, i + 1)
        plt.imshow(xs[idx].squeeze().numpy(), cmap="gray")
        plt.title("t=%d p=%d" % (ys[idx].item(), ps[idx].item()), fontsize=8)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)


save_grid(ok_idx, out_dir / "pred_correct.png", "correct samples")
save_grid(bad_idx, out_dir / "pred_wrong.png", "wrong samples")

with open(out_dir / "acc_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "test_acc", "epochs", "train_size"])
    w.writerow(["SmallCNN", "%.4f" % test_acc, epochs, 8000])

plt.figure()
plt.plot(train_losses)
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.title("week05 cnn train loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "train_loss.png", dpi=120)

plt.figure()
plt.plot(val_accs)
plt.xlabel("epoch")
plt.ylabel("val acc")
plt.title("week05 cnn val acc")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "val_acc.png", dpi=120)

print("结果在", out_dir)
print("第5周完成")
