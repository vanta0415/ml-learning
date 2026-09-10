from pathlib import Path
import csv
import torch
import matplotlib.pyplot as plt

import config
from dataset import get_loaders, set_seed
from model import SmallCNN


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    set_seed(config.SEED)
    device = torch.device("cpu")
    out_dir = Path(__file__).resolve().parent.parent / config.OUT_DIR
    best_path = out_dir / "best_model.pth"

    _, _, test_loader = get_loaders()
    model = SmallCNN().to(device)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            ys.append(yb)
            ps.append(pred)
    y_true = torch.cat(ys)
    y_pred = torch.cat(ps)

    acc = (y_true == y_pred).float().mean().item()
    cm = confusion_matrix(y_true, y_pred)

    with open(out_dir / "test_result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_acc", "best_epoch", "best_val_acc"])
        w.writerow(["%.4f" % acc, ckpt["epoch"], "%.4f" % ckpt["val_acc"]])

    # 保存混淆矩阵数字
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t\\p"] + list(range(10)))
        for i in range(10):
            w.writerow([i] + cm[i].tolist())

    plt.figure(figsize=(6, 5))
    plt.imshow(cm.numpy(), cmap="Blues")
    plt.colorbar()
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)

    print("test_acc=%.3f" % acc)
    print("混淆矩阵已保存")


if __name__ == "__main__":
    main()
