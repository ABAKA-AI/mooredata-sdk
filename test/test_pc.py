#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/1/24 15:31
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_pc.py
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
# # 4

path = 'X:\development\qsy\9.4/'

bags = [
    '2023-09-04-19-04-16_0',
    '2023-09-04-19-04-16_0',
    '2023-09-04-19-09-16_1',
    '2023-09-04-19-09-16_1',
    '2023-09-04-19-14-16_2',
    '2023-09-04-19-14-16_2',
    '2023-09-04-19-19-16_3',
    '2023-09-04-19-19-16_3',
    '2023-09-04-19-24-16_4',
    '2023-09-04-19-29-16_5',
    '2023-09-04-19-39-33_0',
    '2023-09-04-19-44-33_1',
    '2023-09-04-19-44-33_1',
    '2023-09-04-19-54-33_3',
    '2023-09-04-19-54-33_3',
    '2023-09-04-19-59-33_4',
    '2023-09-04-20-04-33_5',
    '2023-09-04-20-04-33_5',
    '2023-09-04-20-09-33_6',
    '2023-09-04-20-14-33_7',
    '2023-09-04-20-19-33_8',
    '2023-09-04-17-36-09_3',
    '2023-09-04-17-41-09_4'
]
clips = [
    '1693825577233512',
    '1693825727233478',
    '1693825847033331',
    '1693825967033453',
    '1693826207033486',
    '1693826327033395',
    '1693826447033517',
    '1693826567033498',
    '1693826717033399',
    '1693827107033529',
    '1693827693933418',
    '1693827933833534',
    '1693828053833408',
    '1693828503833467',
    '1693828743833406',
    '1693828983833491',
    '1693829103833461',
    '1693829223833549',
    '1693829493833480',
    '1693829763833386',
    '1693830003833534',
    '1693820319933365',
    '1693820679933427'
]

head = {
    "FIELDS": ["x", "y", "z", "intensity", 'speed', 'SNR'],
    "SIZE": ["4", "4", "4", "4", '4', '4'],
    "TYPE": ["F", "F", "F", "F", 'F', 'F'],
    "COUNT": ["1", "1", "1", "1", '1', '1']}
for idx, bag in enumerate(tqdm(bags)):
    bins = glob.glob(join(path, bag, clips[idx], 'lidar', 'lidar_top') + '/*.bin')
    for bin in bins:
        if not os.path.exists(join(path, bag, clips[idx], clips[idx], 'lidar_pcd')):
            os.makedirs(join(path, bag, clips[idx], clips[idx], 'lidar_pcd'))
        moore.bin2pcd(bin,
                      join(path, bag, clips[idx], clips[idx], 'lidar_pcd', bin.split('\\')[-1].replace('.bin', '.pcd')),
                      head, 'binary')
