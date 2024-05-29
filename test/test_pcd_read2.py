#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/4/25 16:55
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_pcd_read2.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import moore


pc_points, headers = moore.read_pcd(r"D:\boshi_4D\13109C_downsampled_binary.pcd")
print(pc_points)