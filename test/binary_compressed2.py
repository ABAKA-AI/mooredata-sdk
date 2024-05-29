#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/4/25 17:54
# @Author       : Wu Xinjun
# @Site         : 
# @File         : binary_compressed2.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
import numpy as np
import lzf


def read_binary_compressed_pcd(file_path):
    # 首先，打开文件并读取所有二进制数据
    with open(file_path, 'rb') as f:
        binary_data = f.read()

    # 解析压缩数据和未压缩数据的长度（它们是文件的第一个和第二个32位无符号整数）
    compressed_size = np.frombuffer(binary_data[0:4], dtype=np.uint32)[0]
    decompressed_size = np.frombuffer(binary_data[4:8], dtype=np.uint32)[0]

    # 读取压缩数据
    compressed_data = binary_data[8:8 + compressed_size]

    # 解压缩
    decompressed_data = lzf.decompress(compressed_data, decompressed_size)

    # 返回解压缩的数据
    return decompressed_data

data = read_binary_compressed_pcd(r"X:\development\wxj\datas\boshi_4D\13109C\20240403133805\20240403134243_700\20240403134243.700009_BlindLidar03.pcd")
print(data)