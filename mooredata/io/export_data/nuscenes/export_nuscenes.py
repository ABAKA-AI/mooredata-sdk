#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Optional, Dict
from .nuscenes_tasks.nuscenes_det import nuscenes_detection
from .nuscenes_tasks.nuscenes_seg import nuscenes_segmentation


class ExportNuscenes(nuscenes_detection, nuscenes_segmentation):
    def __init__(self, source_data, output_dir: str, sensor_mapping: Optional[Dict[int, str]] = None):
        """
        Converting mooredata data format to labelme data format
        :param source_data: source mooredata data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportNuscenes, self).__init__(source_data, output_dir, sensor_mapping)

    def moore_json2nuscenes_lidarod(self) -> str:
        """
        将Moore格式JSON转换为NuScenes 3D目标检测格式

        Returns:
            输出目录路径
        """
        # 创建标签映射和类别
        self._create_label_map()
        self._create_categories()
        self._create_nuscenes_attributes()
        self._create_nuscenes_visibility()

        # 处理序列
        self._process_sequences()

        # 更新实例和标注之间的链接
        self._update_instance_annotation_links()

        # 保存NuScenes格式数据
        self._save_nuscenes_od_data()

        return self.output_dir

    def moore_json2nuscenes_lidarseg(self) -> str:
        """
        将Moore格式JSON转换为NuScenes LidarSeg格式

        Returns:
            输出目录路径
        """
        self._create_lidarseg_categories()

        sequences = self.moore_data.get('data', [])

        for seq_idx, sequence in enumerate(sequences):
            info = sequence.get('info', {})
            if 'info' in info:
                info = info.get('info', {})

            num_frames = 0
            if 'pcdUrl' in info:
                num_frames = len(info['pcdUrl'])

            for frame_idx in range(num_frames):
                pcd_url = info['pcdUrl'][frame_idx]
                sample_data_token = self._generate_token()
                self._process_lidarseg_data(sample_data_token, pcd_url, frame_idx, sequence)

        self._save_nuscenes_seg_data()

        return self.output_dir
    