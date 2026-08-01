# 第2周：增强、边缘、形态学 + 轮廓计数小实验
# 运行: python week02_cv_demo.py

from pathlib import Path
import cv2
import numpy as np

data_dir = Path(__file__).parent / "data"
out_dir = Path(__file__).parent / "output_week02"
out_dir.mkdir(exist_ok=True)


def read_img(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def save_img(path, img):
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


def make_coins_img():
    # 自己画几枚“硬币”，方便做计数实验
    img = np.ones((400, 600, 3), dtype=np.uint8) * 230
    centers = [(120, 120), (280, 140), (450, 130), (180, 280), (360, 290)]
    for (x, y) in centers:
        cv2.circle(img, (x, y), 55, (40, 40, 40), -1)
        cv2.circle(img, (x, y), 55, (20, 20, 20), 2)
    path = data_dir / "coins_demo.png"
    save_img(path, img)
    return path


# ---------- 1) 用花图做增强和边缘 ----------
flower = data_dir / "flower1.png"
img = read_img(flower)
if img is None:
    raise RuntimeError("读不到 flower1.png")

# 缩小一点，处理快一些
h, w = img.shape[:2]
img = cv2.resize(img, (w // 2, h // 2))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 亮度对比度：new = gray * alpha + beta
bright = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)

# 直方图均衡
eq = cv2.equalizeHist(gray)

# 滤波去噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel / Canny
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
canny = cv2.Canny(blur, 80, 160)

# 形态学
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
erode = cv2.erode(canny, kernel, iterations=1)
dilate = cv2.dilate(canny, kernel, iterations=1)
opened = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

save_img(out_dir / "flower_bright.png", bright)
save_img(out_dir / "flower_equalize.png", eq)
save_img(out_dir / "flower_blur.png", blur)
save_img(out_dir / "flower_sobel.png", sobel)
save_img(out_dir / "flower_canny.png", canny)
save_img(out_dir / "flower_erode.png", erode)
save_img(out_dir / "flower_dilate.png", dilate)
save_img(out_dir / "flower_open.png", opened)
save_img(out_dir / "flower_close.png", closed)
print("花图增强/边缘/形态学结果已保存")

# ---------- 2) 小实验：硬币轮廓计数 ----------
coins_path = make_coins_img()
coins = read_img(coins_path)
coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
coins_blur = cv2.GaussianBlur(coins_gray, (5, 5), 0)

# 阈值：把深色圆和背景分开
_, binary = cv2.threshold(coins_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 闭运算，把圆补完整一点
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
binary2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2, iterations=2)

contours, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤太小的噪点
good = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 800:
        good.append(cnt)

result = coins.copy()
cv2.drawContours(result, good, -1, (0, 0, 255), 2)
for i, cnt in enumerate(good, 1):
    x, y, ww, hh = cv2.boundingRect(cnt)
    cv2.putText(result, str(i), (x + ww // 2 - 10, y + hh // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

save_img(out_dir / "coins_binary.png", binary2)
save_img(out_dir / "coins_count_ok.png", result)
print("硬币计数（较好结果）：检测到", len(good), "个")

# ---------- 3) 失败样例：阈值设太离谱 ----------
_, bad_bin = cv2.threshold(coins_blur, 250, 255, cv2.THRESH_BINARY_INV)
bad_contours, _ = cv2.findContours(bad_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bad_result = coins.copy()
cv2.drawContours(bad_result, bad_contours, -1, (255, 0, 0), 2)
save_img(out_dir / "coins_count_bad.png", bad_result)
print("失败样例：阈值乱设，轮廓数=", len(bad_contours))

print("\n结果在 output_week02 文件夹")
print("小实验：轮廓提取 + 硬币计数")
