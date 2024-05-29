#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2023/12/26 01:34
# @Author       : Wu Xinjun
# @Site         :
# @File         : test.py
# @Project      : molar
# @Software     : PyCharm
# @Description  :
"""
import  moore
from pathlib import Path

import numpy as np

# data_path = "D:\sdktest\coco\labels\车辆目标检测.json"
# a = Path(data_path).parts[-1]
# print(a)
# bin = r"X:\development\qsy\9.4\2023-09-04-13-42-56_0\1693806416933580\lidar\lidar_top\1693806416933580.bin"
#
# print(np.fromfile(bin).reshape(-1, 6))

a = moore.s2bytes('65fbdfb0b9b4f9cd38e94683')
print(a)