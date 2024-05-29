#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/3/20 17:06
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_pc3.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import glob
import os
import time
from os.path import join

from tqdm import tqdm

import moore
import open3d as o3d
from pypcd import pypcd
from pyntcloud import PyntCloud

# start = time.time()
# pcd = r"D:\volvo\dense_reconstruction.pcd"
# out = r"D:\volvo\dense_reconstruction_binary.pcd"
#
# moore.pcd_ascii2binary(pcd, out)
# end = time.time()
# print(end - start)

# 127
# 97
# 88
# 52

# start = time.time()
# pcd_path = r"D:\volvo\dense_reconstruction.pcd"
# pcd = o3d.io.read_point_cloud(pcd_path)
#
# end = time.time()
# print(end - start)
# 22

# start = time.time()
# pcd_path = r"D:\volvo\dense_reconstruction.pcd"
# # pcd = r"X:\development\wxj\datas\volvo\dense_reconstruction_binary.pcd"
#
# pcd = pypcd.PointCloud.from_path(pcd_path)
# end = time.time()
# print(end - start)
# 22

# start = time.time()
# pcd_path = r"D:\volvo\dense_reconstruction.pcd"
# # pcd = r"X:\development\wxj\datas\volvo\dense_reconstruction_binary.pcd"
#
# pcd = PyntCloud.from_file(pcd_path)
# end = time.time()
# print(end - start)
# 23


# start = time.time()
# pcd = r"D:\volvo\dense_reconstruction.pcd"
# # pcd = r"X:\development\wxj\datas\volvo\dense_reconstruction_binary.pcd"
#
# points, headers = moore.read_pcd(pcd)
# print(points, headers)
# end = time.time()
# print(end - start)
# 4

points = [
          -87.03111065103728,
          23.52644876953514,
          -0.8427688668317186,
          0,
          0,
          2.7344065968461884,
          4.225594951972519,
          2.0722054274616735,
          1.4868380667956134
        ]

corners = moore.box2corners(points)
print(corners)