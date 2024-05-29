#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/4/25 16:02
# @Author       : Wu Xinjun
# @Site         : 
# @File         : binary_compressed.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import struct
import gzip
import time

import numpy as np
import zlib
from pypcd.pypcd import PointCloud
import lzf


# def read_pcd(file_path):
#     with open(file_path, 'rb') as f:
#         line = f.readline().decode().strip()
#         while not line.startswith('DATA'):
#             if line.startswith('POINTS'):
#                 num_points = int(line.split()[-1])
#             line = f.readline().decode().strip()
#
#         data_type, compression_type = map(str.lower, line.split()[-1].split('_'))
#
#         if data_type == 'binary' and compression_type == 'compressed':
#             compressed_data = f.read()
#             expected_size = num_points * 3 * np.dtype(np.float32).itemsize  # x, y, z each of float32
#             decompressed_data = lzf.decompress(compressed_data, expected_size)
#             return np.frombuffer(decompressed_data, dtype=np.float32).reshape(-1, 3)
#         else:
#             print('Unsupported format: %s' % line)

# points = read_pcd(r"X:\development\wxj\datas\boshi_4D\13109C\20240403133805\20240403134243_700\20240403134243.700009_BlindLidar03.pcd")
# print(points)


start = time.time()
single_frame = r"X:\development\wxj\datas\boshi_4D\13109C\20240403133805\20240403134243_700\20240403134243.700009_BlindLidar03.pcd"
points = PointCloud.from_path(single_frame).pc_data
print(points)

end = time.time()
print(end - start)