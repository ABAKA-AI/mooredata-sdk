#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ...factory.data_factory import ImportFactory


class Import(object):
    import_f = ImportFactory()

    @classmethod
    def nuscenes2mooredata_det(cls, nuscenes_root, output_json_path, oss_root, predata=False, json_filename=None, lidar_name=None):
        cls.import_f.import_nuscenes_product(nuscenes_root, output_json_path, oss_root, predata, json_filename, lidar_name).nuscenes2mooredata_det()

    @classmethod
    def nuscenes2mooredata_lane(cls, nuscenes_root, output_json_path, oss_root, predata=False, json_filename=None, lidar_name=None, bin_col=4, intensity_idx=3):
        cls.import_f.import_nuscenes_product(nuscenes_root, output_json_path, oss_root, predata, json_filename, lidar_name, bin_col, intensity_idx).nuscenes2mooredata_lane()
