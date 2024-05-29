#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2023/11/22 17:52
# @Author       : Wu Xinjun
# @Site         : 
# @File         : check.py
# @Project      : MooreSDK
# @Software     : PyCharm
# @Description  : 
"""
from ..moore_data import MOORE


class CheckData():
    def __init__(self, source_data):
        self.source_data = MOORE(source_data)