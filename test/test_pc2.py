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
    '2023-09-04-13-42-56_0',
    '2023-09-04-13-47-56_1',
    '2023-09-04-13-52-56_2',
    '2023-09-04-13-52-56_2',
    '2023-09-04-15-04-37_1',
    '2023-09-04-16-45-47_4',
    '2023-09-04-16-55-47_6',
    '2023-09-04-16-55-47_6',
    '2023-09-04-17-00-47_7',
    '2023-09-04-17-10-47_9',
    '2023-09-04-17-10-47_9',
    '2023-09-04-17-15-47_10',
    '2023-09-04-17-15-47_10',
    '2023-09-04-17-26-09_1',
    '2023-09-04-17-26-09_1',
    '2023-09-04-17-46-09_5',
    '2023-09-04-17-51-09_6',
    '2023-09-04-17-56-09_7',
    '2023-09-04-18-24-31_4',
    '2023-09-04-18-36-19_0'
]
clips = [
    '1693806416933580',
    '1693806566833544',
    '1693806806833446',
    '1693806926833495',
    '1693811198033447',
    '1693817387533246',
    '1693817837433528',
    '1693818017433510',
    '1693818257433445',
    '1693818647433494',
    '1693818797433400',
    '1693818977433471',
    '1693819127433451',
    '1693819599933625',
    '1693819749933557',
    '1693820799933432',
    '1693821159933459',
    '1693821519933533',
    '1693823221733501',
    '1693824050433521'
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
