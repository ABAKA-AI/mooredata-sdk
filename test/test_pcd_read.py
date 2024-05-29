#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/4/25 16:48
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_pcd_read.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import time

import moore


start = time.time()
pc_points, headers = moore.read_pcd(r"D:\boshi_4D\12954C_downsampled_binary.pcd")
print(pc_points)
end = time.time()
print(end - start)