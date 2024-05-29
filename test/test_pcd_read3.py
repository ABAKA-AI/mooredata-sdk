#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/4/25 18:04
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_pcd_read3.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import time
from pypcd.pypcd import PointCloud
import moore


start = time.time()
pc_points, headers = moore.read_pcd(r"X:\development\wxj\datas\boshi_4D\13109C\20240403133805\20240403134243_700\20240403134243.701202_MainLidar01.pcd")
print(pc_points)
end = time.time()
print(end - start)

start1 = time.time()
single_frame = r"X:\development\wxj\datas\boshi_4D\13109C\20240403133805\20240403134243_700\20240403134243.701202_MainLidar01.pcd"
points = PointCloud.from_path(single_frame).pc_data
print(points)

end1 = time.time()
print(end1 - start1)

