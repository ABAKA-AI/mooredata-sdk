#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/3/12 17:57
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test_post_process.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
from moore import PostProcess


# data_path = "D:\sdktest\coco\labels\labels\IAT_test.json"
# out_path = 'D:\sdktest\coco'
# test_size=0.2
# train_size=0.6
# shuffle=True
# PostProcess.coco_split(data_path, out_path, test_size, train_size, shuffle)

data_path = 'D:\sdktest\coco\labels\labels'
out_path = 'D:\sdktest\coco'
merged_file_name = '1'
PostProcess.coco_merge(data_path, out_path, merged_file_name)