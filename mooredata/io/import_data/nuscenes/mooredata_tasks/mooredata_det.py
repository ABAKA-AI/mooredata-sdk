#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from os.path import join
from typing import Dict, List
from pathlib import Path
import numpy as np
from .....utils.pc_tools import quaternion_to_euler


class MooredataDet:
    """
    NuScenes数据集转换工具类，实现将NuScenes格式数据转换为MooreData JSON格式
    """

    def __init__(self, nuscenes_root: str, output_dir: str, oss_root: str, predata: bool = False,
                 json_file_name: str = None, lidar_name: str = None):
        """
        初始化NuScenes数据集转换工具

        Args:
            nuscenes_root: NuScenes数据集根目录路径
        """
        self.nuscenes_root = nuscenes_root
        if json_file_name:
            self.nuscenes_json_path = join(nuscenes_root, json_file_name)
        else:
            self.nuscenes_json_path = join(nuscenes_root, "v1.0-trainval")
        self.output_dir = output_dir
        self.oss_root = oss_root
        self.predata = predata
        self.lidar_name = lidar_name
        self.scenes_data = []

    def _load_json_file(self, filename: str) -> Dict:
        """
        加载JSON文件

        Args:
            filename: JSON文件名

        Returns:
            解析后的JSON数据
        """
        file_path = join(self.nuscenes_json_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_scene_sample_tokens(self, scene_token: str) -> List[str]:
        """
        获取场景对应的所有sample tokens

        Args:
            scene_token: 场景token

        Returns:
            该场景下的所有sample tokens
        """
        scenes = self._load_json_file("scene.json")
        scene = next((s for s in scenes if s['token'] == scene_token), None)
        if not scene:
            return []

        samples = self._load_json_file("sample.json")
        sample_tokens = []
        current_token = scene['first_sample_token']

        while current_token:
            sample = next((s for s in samples if s['token'] == current_token), None)
            if not sample:
                break
            sample_tokens.append(current_token)
            current_token = sample['next']

        return sample_tokens

    def _get_key_frame_data(self, sample_tokens: List[str], data_type: str) -> List[Dict]:
        """
        获取关键帧数据

        Args:
            sample_tokens: sample tokens列表
            data_type: 数据类型('pointcloud'或'image')

        Returns:
            关键帧数据列表
        """
        sample_data = self._load_json_file("sample_data.json")
        key_frames = []

        for token in sample_tokens:
            matched_data = [d for d in sample_data
                            if d['sample_token'] == token
                            and d.get('is_key_frame', False)]

            if data_type == 'pointcloud':
                matched_data = [d for d in matched_data if self.lidar_name in d['filename']]
                key_frames.append(matched_data[0])
            elif data_type == 'image':
                matched_data = [d for d in matched_data if d['fileformat'] in ['jpg', 'jpeg', 'png']]
                key_frames = matched_data

        return key_frames

    def get_pointcloud_paths(self, scene_token: str) -> List[str]:
        """
        获取场景的点云文件路径

        Args:
            scene_token: 场景token

        Returns:
            点云文件路径列表
        """
        sample_tokens = self._get_scene_sample_tokens(scene_token)
        key_frames = self._get_key_frame_data(sample_tokens, 'pointcloud')
        return [str(join(self.oss_root, f['filename'])) for f in key_frames]

    def get_image_paths(self, scene_token: str) -> List[List[str]]:
        """
        获取场景的图像文件路径

        Args:
            scene_token: 场景token

        Returns:
            图像文件路径二维数组
        """
        sample_tokens = self._get_scene_sample_tokens(scene_token)

        # 加载sensor和calibrated_sensor数据
        sensors = self._load_json_file("sensor.json")
        calibrated_sensors = self._load_json_file("calibrated_sensor.json")

        # 创建相机token到校准传感器token的映射
        camera_to_calibrated = {
            cs['token']: cs['sensor_token']
            for cs in calibrated_sensors
            if cs['sensor_token'] in {s['token'] for s in sensors if s['modality'] == 'camera'}
        }

        # 获取相机token到channel的映射
        camera_order = {s['token']: idx for idx, s in enumerate(sensors)
                        if s['modality'] == 'camera'}

        # 按sample_tokens顺序处理数据
        images = []
        for token in sample_tokens:
            # 获取当前token对应的所有关键帧数据
            frame_data = self._get_key_frame_data([token], 'image')
            if frame_data:
                sorted_images = sorted(
                    frame_data,
                    key=lambda x: camera_order.get(camera_to_calibrated.get(x['calibrated_sensor_token'], ''),
                                                   float('inf'))
                )
                image_paths = [str(join(self.oss_root, d['filename'])) for d in sorted_images]
                images.append(image_paths)

        return images

    def get_pose_data(self, scene_token: str) -> List[Dict]:
        """
        获取场景的位姿数据(仅返回与点云数据对应的位姿)

        Args:
            scene_token: 场景token

        Returns:
            位姿数据列表
        """
        sample_tokens = self._get_scene_sample_tokens(scene_token)
        key_frames = self._get_key_frame_data(sample_tokens, 'pointcloud')

        pointcloud_data = [{
            'filename': f['filename'],
            'ego_pose_token': f['ego_pose_token']
        } for f in key_frames]

        # 加载ego_pose数据
        ego_poses = self._load_json_file("ego_pose.json")
        pose_map = {pose['token']: pose for pose in ego_poses}

        first_pose_token = pointcloud_data[0]['ego_pose_token']
        first_pose_translation = pose_map[first_pose_token]['translation']
        poses = []
        for idx, data in enumerate(pointcloud_data):
            pose = pose_map.get(data['ego_pose_token'])
            if pose:
                r, p, y = quaternion_to_euler(pose['rotation'][1], pose['rotation'][2],
                                              pose['rotation'][3], pose['rotation'][0])
                poses.append({
                    'name': idx,
                    'posMatrix': (np.array(pose['translation']) - np.array(first_pose_translation)).tolist() + [r, p, y]
                })

        return poses

    def convert_to_mooredata_det(self) -> str:
        """
        将NuScenes数据转换为MooreData JSON格式

        Args:
            output_path: 输出文件路径
            predata: 是否包含preData标签数据

        Returns:
            转换后的文件路径
        """
        scenes = self._load_json_file('scene.json')
        moore_data = []

        for scene in scenes:
            scene_token = scene['token']
            scene_data = {
                "info": {
                    "pcdUrl": self.get_pointcloud_paths(scene_token),
                    "imgUrls": self.get_image_paths(scene_token),
                    "locations": self.get_pose_data(scene_token)
                }
            }

            if self.predata:
                scene_data["preData"] = []
                # 加载sample.json获取sample_token
                sample_file = join(self.nuscenes_json_path, "sample.json")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    samples = json.load(f)

                # 获取该场景下的所有sample_token
                scene_sample_tokens = [s['token'] for s in samples if s['scene_token'] == scene_token]

                # 加载标注数据
                sample_anno_file = join(self.nuscenes_json_path, "sample_annotation.json")
                with open(sample_anno_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)

                for anno in annotations:
                    if anno['sample_token'] in scene_sample_tokens:
                        scene_data["preData"].append({
                            "token": anno['token'],
                            "category": anno['category_name'],
                            "bbox": anno['bbox'],
                            "attributes": anno['attribute_tokens']
                        })

            moore_data.append(scene_data)

        # 保存转换结果
        output_path = Path(self.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(moore_data, f, indent=2, ensure_ascii=False)

        return str(output_path)
