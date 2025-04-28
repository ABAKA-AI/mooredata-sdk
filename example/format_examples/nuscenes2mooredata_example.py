#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mooredata import Import

nuscenes_root = 'X:/XXX/XXX'
json_filename = 'v1.0'
output_dir = 'X:/XXX/XXX.json'
oss_root = 'https://BUCKET/XXX/'
predata=False

# dynamic task import json
Import.nuscenes2mooredata_det(nuscenes_root, output_dir, oss_root, predata=predata, json_filename=json_filename, lidar_name='LIDAR_0')
# 4D task import json
Import.nuscenes2mooredata_lane(nuscenes_root, output_dir, oss_root, predata=predata, json_filename=json_filename, lidar_name='LIDAR_0', bin_col=5, intensity_idx=3)
