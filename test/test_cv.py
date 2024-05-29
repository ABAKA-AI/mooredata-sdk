#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/1/10 22:49
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_cv.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import cv2
from PIL import Image
import moore


# 加载图像
# img = cv2.imread(r'G:\opensource\test\12 (1).jpg')
# print(img)

# flip = moore.flip_image(img)
# cv2.imwrite(r'G:\opensource\test\12.jpg', flip)

# img_flipped = cv2.flip(img, 1)
# cv2.imwrite(r'G:\opensource\test\12_flipped.jpg', img_flipped)

moore.flip_images_in_folder('G:\zhengshu\input\sdk_test\png', 'G:\zhengshu\input\sdk_test\png_out', 1)

moore.rotate_images_in_folder('G:\zhengshu\input\sdk_test\png', 'G:\zhengshu\input\sdk_test\png_out', 90)