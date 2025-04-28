#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .mooredata_tasks.mooredata_det import MooredataDet
from .mooredata_tasks.mooredata_lane import MooredataLane


class ImportNuscenes(MooredataDet, MooredataLane):
    def __init__(self, nuscenes_root, output_json_path, oss_root, predata=False, json_file_name=None, lidar_name: str = None, bin_col=4, intensity_idx=3):
        self.nuscenes_root = nuscenes_root
        self.output_json_path = output_json_path
        self.oss_root = oss_root
        self.predata = predata
        self.json_file_name = json_file_name
        self.lidar_name = lidar_name
        self.bin_col = bin_col
        self.intensity_idx = intensity_idx
        self._det_initialized = False
        self._lane_initialized = False

    def nuscenes2mooredata_det(self):
        """
        将NuScenes数据集转换为MooreData 3D JSON格式
        """
        if not self._det_initialized:
            MooredataDet.__init__(self, self.nuscenes_root, self.output_json_path, self.oss_root, self.predata, self.json_file_name, self.lidar_name)
            self._det_initialized = True
        result = self.convert_to_mooredata_det()
        return result

    def nuscenes2mooredata_lane(self):
        """
        将NuScenes数据集转换为MooreData 4D JSON格式
        """
        if not self._lane_initialized:
            MooredataLane.__init__(self, self.nuscenes_root, self.output_json_path, self.oss_root, self.predata, self.json_file_name, self.lidar_name, self.bin_col, self.intensity_idx)
            self._lane_initialized = True
        result = self.convert_to_mooredata_lane()
        return result

