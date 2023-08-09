import cv2
import numpy as np

# 读取原图和分割结果
original_image = cv2.imread('../img/prostate/0/img.png')
segmented_image = cv2.imread('../img/prostate/0/mask.png')

# 将分割结果转换为灰度图像
print(segmented_image)
print(segmented_image.shape)
print(type(segmented_image))
segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
segmented_gray[segmented_gray!=255]=1
segmented_gray[segmented_gray==255]=0
print(segmented_gray)

# 使用阈值处理将分割结果转换为二值图像
_, segmented_binary = cv2.threshold(segmented_gray, 0, 255, cv2.THRESH_BINARY)
print(segmented_binary.shape)
print(type(segmented_binary))
print(segmented_binary)

# 查找分割区域的轮廓
contours, _ = cv2.findContours(segmented_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原图上绘制分割区域轮廓
print(original_image.shape)
print(type(original_image))
print(np.unique(original_image))
cv2.drawContours(original_image, contours, -1, (0, 255, 0), 1)
print(np.unique(original_image))

# 显示结果图像
cv2.imshow('Segmentation Result', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
