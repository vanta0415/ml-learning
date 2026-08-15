# 第1周：读图、灰度、HSV、缩放

from pathlib import Path
import cv2
import numpy as np

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output_week01"
out_dir.mkdir(exist_ok=True)


def read_img(path):
    # 中文路径直接imread会失败，所以用这种方式读
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def save_img(path, img):
    ext = path.suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


# 读取data里的图片（至少3张）
imgs = []
for f in sorted(data_dir.iterdir()):
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        imgs.append(f)
imgs = imgs[:3]

print("一共读取", len(imgs), "张图")

for i, path in enumerate(imgs, 1):
    img = read_img(path)
    if img is None:
        print("读失败:", path.name)
        continue

    h, w, c = img.shape
    print(f"\n第{i}张: {path.name}")
    print("尺寸:", h, "x", w, "通道:", c)
    print("像素范围:", img.min(), "~", img.max())
    print("左上角BGR:", img[0, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    half = cv2.resize(img, (w // 2, h // 2))

    name = path.stem
    save_img(out_dir / (name + "_gray.png"), gray)
    save_img(out_dir / (name + "_hsv.png"), hsv)
    save_img(out_dir / (name + "_half.png"), half)
    print("已保存灰度/HSV/缩小图")

print("\n结果在 output_week01 文件夹")
