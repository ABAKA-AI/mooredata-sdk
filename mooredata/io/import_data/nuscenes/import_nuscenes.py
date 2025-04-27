#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .nuscenes_tasks.nuscenes_det import NuscenesDet


class ImportNuscenes(NuscenesDet):
    def __init__(self, nuscenes_root, output_dir, oss_root, predata=False, json_file_name=None, is_key_frame=False):

        super(ImportNuscenes, self).__init__(nuscenes_root, output_dir, oss_root, predata, json_file_name, is_key_frame)

    def nuscenes2mooredata_det(self):
        """
        将NuScenes数据集转换为MooreData JSON格式
        """
        # 调用功能函数完成转换
        result = self.convert_to_mooredata()
        return result
        
    def convert_to_other_format(self, format_name):
        """
        预留其他格式转换的接口
        """
        raise NotImplementedError(f"{format_name}格式转换尚未实现")

