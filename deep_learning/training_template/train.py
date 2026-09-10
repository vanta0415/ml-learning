from pathlib import Path
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import config
from dataset import get_loaders, set_seed
from model import SmallCNN


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def main():
    set_seed(config.SEED)
    device = torch.device("cpu")
    out_dir = Path(__file__).resolve().parent.parent / config.OUT_DIR
    out_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders()
    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_acc"])

    train_losses, val_accs = [], []
    best_acc = -1.0
    best_path = out_dir / "best_model.pth"

    print("开始训练...")
    for epoch in range(config.EPOCHS):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        train_loss = total / n
        val_acc = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, "%.4f" % train_loss, "%.4f" % val_acc])

        print("epoch%d loss=%.4f val=%.3f" % (epoch + 1, train_loss, val_acc))

        # 验证最好时保存
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": best_acc,
            }, best_path)
            print("  保存 best_model.pth, val_acc=%.3f" % best_acc)

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("week07 train loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss.png", dpi=120)

    plt.figure()
    plt.plot(val_accs)
    plt.xlabel("epoch")
    plt.ylabel("val acc")
    plt.title("week07 val acc")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "val_acc.png", dpi=120)

    print("训练完成，best_val=%.3f" % best_acc)
    print("日志:", log_path)


if __name__ == "__main__":
    main()
