第1周笔记

日期：2026-07-25
代码：image_processing/week01_image_io.py

1. 这周做了啥

用OpenCV读了3张花的图片，转成灰度图和HSV，又缩放到一半保存。
也打印了每张图的尺寸、通道数和像素值范围。

2. 几个概念

(1) 像素
图片上的小点，每个点对应一些数字。

(2) 通道
彩色图一般是3个通道。OpenCV读出来是BGR。
灰度图只有1个通道。

(3) 灰度图
只有明暗，没有颜色。

(4) BGR和RGB
都是蓝绿红/红绿蓝，只是顺序不同。OpenCV默认BGR。

(5) HSV
H是颜色种类，S是鲜艳程度，V是亮度。

3. 怎么运行

cd image_processing
python week01_image_io.py

4. 结果

输入：flower1.png、flower2.png、flower3.png
输出：每张图各有 _gray、_hsv、_half，一共9张，在output_week01里。

flower1大概1202x928，3通道
flower2大概1280x940，3通道
flower3大概668x500，3通道
像素大概在0到255之间。

5. 遇到的问题

路径有中文时，cv2.imread有时读不了，改成fromfile再imdecode就可以了。
