#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
import json
import os
import random
import zlib
from typing import Optional, Dict
import numpy as np


class nuscenes_segmentation():
    def __init__(self, source_data, output_dir: str, sensor_mapping: Optional[Dict[int, str]] = None):
        self.moore_data = source_data
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), 'nuscenes')
        self.sensor_mapping = sensor_mapping  # 传感器映射

        self.lidarseg_category_map = {}
        self.lidarseg_data = {
            'lidarseg': [],
            'lidarseg_category': [],
            'panoptic': []
        }

        os.makedirs(output_dir, exist_ok=True)

        self.json_dir = os.path.join(output_dir, 'v1.0-trainval')
        self.lidarseg_dir = os.path.join(output_dir, 'lidarseg/v1.0-trainval')
        self.panoptic_dir = os.path.join(output_dir, 'panoptic/v1.0-trainval')

        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.lidarseg_dir, exist_ok=True)
        os.makedirs(self.panoptic_dir, exist_ok=True)

    def _generate_token(self):
        token = ''.join([format(random.randint(0, 15), 'x') for _ in range(32)])
        return token

    def _create_lidarseg_categories(self):
        """创建LidarSeg类别数据"""
        label_configs = self.moore_data['task']['setting']['labelConfig']

        for idx, label_config in enumerate(label_configs):
            if 'label' in label_config:
                label_name = label_config['label']
                label_key = label_config.get('key', f'label_{idx}')
                category_id = idx + 1
                category = {
                    'token': self._generate_token(),
                    'name': label_name,
                    'description': f"Category {label_name}",
                    'index': category_id,
                    'color': label_config.get('color', '#FFFFFF')
                }

                self.lidarseg_data['lidarseg_category'].append(category)
                self.lidarseg_category_map[label_key] = category_id
        print(f"创建了 {len(self.lidarseg_data['lidarseg_category'])} 个LidarSeg类别")

    def _process_panoptic_data(self, sample_data_token: str, pcd_url: str, point_labels: np.ndarray):
        """
        处理全景分割数据

        Args:
            sample_data_token: 对应的sample_data的token
            pcd_url: 点云文件URL
            frame_idx: 帧索引
            point_labels: 点云标签数组
        """
        panoptic_filename = f"{os.path.basename(pcd_url).split('.')[0]}.npz"
        panoptic_path = os.path.join(self.panoptic_dir, panoptic_filename)

        instance_ids = np.zeros(len(point_labels), dtype=np.uint16)

        # TODO: 根据3D框信息将点分配给实例

        os.makedirs(os.path.dirname(panoptic_path), exist_ok=True)
        np.savez_compressed(
            panoptic_path,
            semantics=point_labels.astype(np.uint8),
            instance=instance_ids
        )

        panoptic_entry = {
            'token': sample_data_token,
            'sample_data_token': sample_data_token,
            'filename': f"panoptic/v1.0-trainval/{panoptic_filename}"
        }
        self.lidarseg_data['panoptic'].append(panoptic_entry)

        print(f"保存panoptic文件: {panoptic_path}")

    def _process_lidarseg_data(self, sample_data_token: str, pcd_url: str, frame_idx: int, sequence: Dict):
        """
        处理LidarSeg数据

        Args:
            sample_data_token: 对应的sample_data的token
            pcd_url: 点云文件URL
            frame_idx: 帧索引
            sequence: 序列数据
        """
        semantic_label = None
        if 'labels' in sequence:
            for label_item in sequence.get('labels', []):
                label_data = label_item.get('data', {})

                if (label_data.get('frameIndex') == frame_idx and
                        label_data.get('drawType') == 'SEMANTIC_BASE' and
                        'pLabelIdMap' in label_data):
                    semantic_label = label_data
                    break

        if not semantic_label:
            print(f"帧 {frame_idx} 未找到语义分割标签")
            return

        plabelidmap = semantic_label.get('pLabelIdMap', '')

        if not plabelidmap:
            print(f"帧 {frame_idx} 的语义分割标签中未找到pLabelIdMap")
            return

        try:
            decompressed_data = zlib.decompress(
                base64.b64decode(plabelidmap), 16 + zlib.MAX_WBITS
            )

            point_labels = np.frombuffer(decompressed_data, dtype=np.uint8)

            if len(point_labels) == 0:
                print(f"帧 {frame_idx} 的pLabelIdMap解析结果为空")
                return

            print(f"帧 {frame_idx} 的pLabelIdMap解析成功，共 {len(point_labels)} 个点")

            lidarseg_filename = f"{os.path.basename(pcd_url).split('.')[0]}.bin"
            lidarseg_path = os.path.join(self.lidarseg_dir, lidarseg_filename)

            mapped_labels = np.zeros(len(point_labels), dtype=np.uint8)

            label_mapping = {}
            if 'task' in self.moore_data and 'setting' in self.moore_data['task'] and 'labelConfig' in \
                    self.moore_data['task']['setting']:
                label_configs = self.moore_data['task']['setting']['labelConfig']
                for idx, config in enumerate(label_configs):
                    label_key = config.get('key', f'label_{idx}')
                    label_id = idx + 1
                    label_mapping[idx] = label_id
                    self.lidarseg_category_map[label_key] = label_id

            for i, label_id in enumerate(point_labels):
                if label_id in label_mapping:
                    mapped_labels[i] = label_mapping[label_id]
                else:
                    mapped_labels[i] = 0

            os.makedirs(os.path.dirname(lidarseg_path), exist_ok=True)
            mapped_labels.astype(np.uint8).tofile(lidarseg_path)

            lidarseg_entry = {
                'token': sample_data_token,
                'sample_data_token': sample_data_token,
                'filename': f"lidarseg/v1.0-trainval/{lidarseg_filename}"
            }
            self.lidarseg_data['lidarseg'].append(lidarseg_entry)

            print(f"保存lidarseg文件: {lidarseg_path}")

            self._process_panoptic_data(sample_data_token, pcd_url, point_labels)

        except Exception as e:
            print(f"处理帧 {frame_idx} 的pLabelIdMap时出错: {e}")
            return

    def _save_nuscenes_seg_data(self):
        for key in self.lidarseg_data.keys():
            output_file = os.path.join(self.json_dir, f"{key}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.lidarseg_data[key], f, ensure_ascii=False, indent=2)

            print(f"已保存: {output_file}")
