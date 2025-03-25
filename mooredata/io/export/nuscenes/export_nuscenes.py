#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import base64
import zlib
from mooredata.utils.pc_tools import euler_to_quaternion
from mooredata.utils.general import download_file


class ExportNuscenes:
    """
    将Moore格式数据转换为NuScenes格式
    支持3D目标检测(3DOD)和LidarSeg格式
    """
    
    def __init__(self, source_data, output_dir: str, sensor_mapping: Optional[Dict[int, str]] = None):
        """
        初始化转换器
        
        Args:
            source_data: Moore格式数据
            output_dir: 输出NuScenes格式数据的目录
            sensor_mapping: 传感器映射，键为传感器索引，值为传感器名称
                           如果为None，则从Moore格式数据中获取
        """
        self.moore_data = source_data
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), 'nuscenes')
        self.label_map = {}  # 标签映射
        self.attribute_map = {}  # 属性映射
        self.camera_sensors = []  # 相机传感器列表
        self.sample_data_tokens = {}  # 用于跟踪每个通道的样本数据token
        self.sensor_mapping = sensor_mapping  # 传感器映射
        self.category_mapping = {}  # 类别映射
        if self.sensor_mapping is None:
            self._get_sensor_mapping_from_moore()
        self.lidarseg_category_map = {}
        self.nuscenes_data = {
            'category': [],
            'attribute': [],
            'visibility': [],
            'instance': [],
            'sensor': [],
            'calibrated_sensor': [],
            'ego_pose': [],
            'log': [],
            'scene': [],
            'sample': [],
            'sample_data': [],
            'sample_annotation': [],
            'map': [],
        }
        
        self.lidarseg_data = {
            'lidarseg': [],
            'lidarseg_category': [],
            'panoptic': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.samples_dir = os.path.join(output_dir, 'samples')
        self.sweeps_dir = os.path.join(output_dir, 'sweeps')
        self.maps_dir = os.path.join(output_dir, 'maps')
        self.json_dir = os.path.join(output_dir, 'v1.0-trainval')
        self.lidarseg_dir = os.path.join(output_dir, 'lidarseg/v1.0-trainval')
        self.panoptic_dir = os.path.join(output_dir, 'panoptic/v1.0-trainval')
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.sweeps_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.lidarseg_dir, exist_ok=True)
        os.makedirs(self.panoptic_dir, exist_ok=True)
        
    def _generate_token(self) -> str:
        """生成唯一的token"""
        return str(uuid.uuid4())
    
    def _save_nuscenes_od_data(self):
        """
        保存NuScenes格式数据
        
        Args:
            include_lidarseg: 是否包含lidarseg数据
        """
        for key in self.nuscenes_data.keys():
            output_file = os.path.join(self.json_dir, f"{key}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.nuscenes_data[key], f, ensure_ascii=False, indent=2)
            
            print(f"已保存: {output_file}")

    def _save_nuscenes_seg_data(self):
        for key in self.lidarseg_data.keys():
            output_file = os.path.join(self.json_dir, f"{key}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.lidarseg_data[key], f, ensure_ascii=False, indent=2)

            print(f"已保存: {output_file}")
    
    def _get_sensor_mapping_from_moore(self):
        """从Moore格式JSON中获取传感器映射"""
        self.sensor_mapping = {}
        
        fusion_config = []
        if 'task' in self.moore_data and 'setting' in self.moore_data['task']:
            if 'toolSetting' in self.moore_data['task']['setting']:
                if 'fusionConfig' in self.moore_data['task']['setting']['toolSetting']:
                    fusion_config = self.moore_data['task']['setting']['toolSetting']['fusionConfig']
        
        num_cameras = 0
        if 'data' in self.moore_data and len(self.moore_data['data']) > 0:
            if 'info' in self.moore_data['data'][0] and 'info' in self.moore_data['data'][0]['info']:
                if 'imgUrls' in self.moore_data['data'][0]['info']['info'] and len(self.moore_data['data'][0]['info']['info']['imgUrls']) > 0:
                    num_cameras = len(self.moore_data['data'][0]['info']['info']['imgUrls'][0])
        
        if fusion_config and len(fusion_config) == num_cameras:
            for i, config in enumerate(fusion_config):
                sensor_name = config.get('name', f'CAM_{i+1}')
                self.sensor_mapping[i] = sensor_name
                self.camera_sensors.append(sensor_name)
        else:
            for i in range(num_cameras):
                sensor_name = f'CAM_{i+1}'
                
                self.sensor_mapping[i] = sensor_name
                self.camera_sensors.append(sensor_name)
        
        print(f"传感器映射: {self.sensor_mapping}")
    
    def _create_nuscenes_sensors(self):
        """创建NuScenes传感器数据"""
        lidar_sensor = {
            'token': self._generate_token(),
            'channel': 'LIDAR_TOP',
            'modality': 'lidar'
        }
        self.nuscenes_data['sensor'].append(lidar_sensor)
        self.lidar_sensor_token = lidar_sensor['token']
        
        self.camera_sensor_tokens = {}
        for idx, sensor_name in self.sensor_mapping.items():
            camera_sensor_token = self._generate_token()
            camera_sensor = {
                'token': camera_sensor_token,
                'channel': sensor_name,
                'modality': 'camera'
            }
            self.nuscenes_data['sensor'].append(camera_sensor)
            self.camera_sensor_tokens[idx] = camera_sensor_token
    
    def _create_label_map(self):
        """创建标签映射"""
        label_configs = self.moore_data['task']['setting']['labelConfig']
        
        for idx, config in enumerate(label_configs):
            label_name = config['label']
            label_key = config.get('key', f'label_{idx}')
            
            english_name = label_name
            if 'labelAlias' in self.moore_data['task']['setting']:
                for alias in self.moore_data['task']['setting']['labelAlias']:
                    if label_key in alias:
                        english_name = alias[label_key].get('label', label_name)
            
            self.label_map[label_key] = {
                'name': label_name,
                'english_name': english_name,
                'color': config.get('color', '#FFFFFF'),
                'attributes': []
            }
            
            if 'attributes' in config:
                for attr_idx, attr in enumerate(config['attributes']):
                    attr_name = attr['label']
                    attr_key = f"{label_key}_{attr_idx}"
                    
                    self.attribute_map[attr_key] = {
                        'name': attr_name,
                        'values': []
                    }
                    
                    if 'children' in attr:
                        for child_idx, child in enumerate(attr['children']):
                            child_name = child['label']
                            child_key = f"{attr_key}_{child_idx}"
                            
                            self.attribute_map[attr_key]['values'].append({
                                'key': child_key,
                                'name': child_name
                            })
                            
                    self.label_map[label_key]['attributes'].append(attr_key)
    
    def _create_categories(self):
        """创建类别数据"""
        label_configs = self.moore_data['task']['setting']['labelConfig']
        
        for label_config in label_configs:
            if 'label' in label_config:
                label_name = label_config['label']
                
                category_token = self._generate_token()
                category = {
                    'token': category_token,
                    'name': label_name,
                    'description': label_name
                }
                
                self.nuscenes_data['category'].append(category)
                
                self.category_mapping[label_name] = category_token
        
        print(f"创建了 {len(self.nuscenes_data['category'])} 个类别")
    
    def _create_nuscenes_attributes(self):
        """创建NuScenes属性数据"""
        if 'task' in self.moore_data and 'setting' in self.moore_data['task'] and 'labelConfig' in self.moore_data['task']['setting']:
            label_configs = self.moore_data['task']['setting']['labelConfig']
            
            for label_config in label_configs:
                label_name = label_config.get('label', '')
                attributes = label_config.get('attributes', [])
                
                self._generate_attribute_combinations(label_name, attributes, [])
        
        if not self.nuscenes_data['attribute']:
            self._create_default_attributes()
        
        print(f"创建了 {len(self.nuscenes_data['attribute'])} 个属性")
    
    def _generate_attribute_combinations(self, label_name, attributes, current_path):
        """生成所有可能的属性组合"""
        if not attributes:
            if current_path:
                attr_name = f"{label_name}." + ".".join(current_path)
                attr_token = self._generate_token()
                attr = {
                    'token': attr_token,
                    'name': attr_name,
                    'description': f"Attribute {attr_name}"
                }
                self.nuscenes_data['attribute'].append(attr)
                self.attribute_map[attr_name] = attr_token
            return
        
        current_attr = attributes[0]
        attr_label = current_attr.get('label', '')
        children = current_attr.get('children', [])
        
        if not children:
            self._generate_attribute_combinations(label_name, attributes[1:], current_path)
            return
        
        for child in children:
            child_label = child.get('label', '')
            new_path = current_path + [child_label]
            
            if len(attributes) > 1:
                self._generate_attribute_combinations(label_name, attributes[1:], new_path)
            else:
                attr_name = f"{label_name}." + ".".join(new_path)
                attr_token = self._generate_token()
                attr = {
                    'token': attr_token,
                    'name': attr_name,
                    'description': f"Attribute {attr_name}"
                }
                self.nuscenes_data['attribute'].append(attr)
                self.attribute_map[attr_name] = attr_token
    
    def _create_default_attributes(self):
        """创建默认的NuScenes属性"""
        default_attributes = [
            'vehicle.moving', 'vehicle.parked', 'vehicle.stopped',
            'cycle.with_rider', 'cycle.without_rider',
            'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting'
        ]
        
        for attr_name in default_attributes:
            attr_token = self._generate_token()
            attr = {
                'token': attr_token,
                'name': attr_name,
                'description': f"Attribute {attr_name}"
            }
            self.nuscenes_data['attribute'].append(attr)
            self.attribute_map[attr_name] = attr_token
    
    def _create_nuscenes_visibility(self):
        """创建NuScenes可见性数据"""
        visibility_levels = [
            {'level': "v0-40", 'token': '1', 'description': '0-40% 可见'},
            {'level': "v40-60", 'token': '2', 'description': '40-60% 可见'},
            {'level': "v60-80", 'token': '3', 'description': '60-80% 可见'},
            {'level': "v80-100", 'token': '4', 'description': '80-100% 可见'}
        ]
        
        for vis in visibility_levels:
            visibility = {
                'token': vis['token'],
                'level': vis['level'],
                'description': vis['description']
            }
            self.nuscenes_data['visibility'].append(visibility)
    
    def _process_sequences(self):
        """处理所有序列数据"""
        self._create_nuscenes_sensors()
        
        sequences = self.moore_data.get('data', [])
        
        for seq_idx, sequence in enumerate(sequences):
            self._process_single_sequence(seq_idx, sequence)
    
    def _process_single_sequence(self, seq_idx: int, sequence: Dict):
        """处理单个序列数据"""
        log_token = self._generate_token()
        log = {
            'token': log_token,
            'logfile': f"log_{seq_idx}",
            'vehicle': 'vehicle',
            'date_captured': datetime.now().strftime('%Y-%m-%d'),
            'location': 'location'
        }
        self.nuscenes_data['log'].append(log)

        map_token = self._generate_token()
        map_data = {
            'token': map_token,
            'log_tokens': [log_token],
            'filename': '',
            'category': 'semantic_prior'
        }
        self.nuscenes_data['map'].append(map_data)
    
        scene_token = self._generate_token()
        scene = {
            'token': scene_token,
            'name': f"scene_{seq_idx}",
            'description': f"Scene {seq_idx} from Moore data",
            'log_token': log_token,
            'nbr_samples': 0,
            'first_sample_token': None,
            'last_sample_token': None
        }
        self.nuscenes_data['scene'].append(scene)
        
        info = sequence.get('info', {})
        if 'info' in info:
            info = info.get('info', {})
        
        num_frames = 0
        if 'pcdUrl' in info:
            num_frames = len(info['pcdUrl'])
        
        prev_sample_token = None
        first_sample_token = None
        
        for frame_idx in range(num_frames):
            translation = [0, 0, 0]
            rotation = [1, 0, 0, 0]
            
            if 'locations' in info and frame_idx < len(info['locations']):
                location = info['locations'][frame_idx]
                if 'posMatrix' in location and len(location['posMatrix']) >= 6:
                    pos_matrix = location['posMatrix']
                    translation = [pos_matrix[0], pos_matrix[1], pos_matrix[2]]
                    
                    euler_angles = [
                        pos_matrix[3],
                        pos_matrix[4],
                        pos_matrix[5]
                    ]
                    rotation = euler_to_quaternion(euler_angles)

            ego_pose_token = self._generate_token()
            ego_pose = {
                'token': ego_pose_token,
                'translation': translation,
                'rotation': rotation,
                'timestamp': int(datetime.now().timestamp() * 1000000) + frame_idx * 100000
            }
            self.nuscenes_data['ego_pose'].append(ego_pose)
            
            sample_token = self._generate_token()
            sample = {
                'token': sample_token,
                'timestamp': ego_pose['timestamp'],
                'scene_token': scene_token,
                'prev': prev_sample_token if prev_sample_token else "",
                'next': "",
                'data': {}
            }
            
            if prev_sample_token:
                for prev_sample in self.nuscenes_data['sample']:
                    if prev_sample['token'] == prev_sample_token:
                        prev_sample['next'] = sample_token
                        break
            else:
                first_sample_token = sample_token
            
            self._process_lidar_data(sample, self.lidar_sensor_token, info['pcdUrl'][frame_idx], ego_pose_token, frame_idx)
            
            if frame_idx < len(info.get('imgUrls', [])):
                img_urls = info['imgUrls'][frame_idx]
                for camera_idx, img_url in enumerate(img_urls):
                    sensor_token = self.camera_sensor_tokens.get(camera_idx)
                    if sensor_token:
                        self._process_camera_data(sample, sensor_token, img_url, ego_pose_token, frame_idx, camera_idx)
            
            if 'labels' in sequence:
                self._process_annotations(sample, sequence, frame_idx)
            
            self.nuscenes_data['sample'].append(sample)
            prev_sample_token = sample_token
        
        scene['nbr_samples'] = num_frames
        scene['first_sample_token'] = first_sample_token
        scene['last_sample_token'] = prev_sample_token
    
    def _process_lidar_data(self, sample: Dict, sensor_token: str, pcd_url: str, ego_pose_token: str, frame_idx: int):
        """处理激光雷达数据"""
        calibrated_sensor_token = self._generate_token()
        
        translation = [0, 0, 0]
        rotation = [1, 0, 0, 0]

        try:
            for sequence in self.moore_data.get('data', []):
                info = sequence.get('info', {})

                if frame_idx < len(info['info']['locations']) and frame_idx < len(info['info']['pcdUrl']):
                    location = info['info']['locations'][frame_idx]
                    
                    if 'posMatrix' in location:
                        pos_matrix = location['posMatrix']

                        translation = [pos_matrix[0], pos_matrix[1], pos_matrix[2]]
                        
                        euler_angles = [
                            pos_matrix[3],
                            pos_matrix[4],
                            pos_matrix[5]
                        ]
                        rotation = euler_to_quaternion(euler_angles)
                        
                        break
        except Exception as e:
            print(f"获取位置矩阵失败: {e}")
        
        calibrated_sensor = {
            'token': calibrated_sensor_token,
            'sensor_token': sensor_token,
            'translation': translation,
            'rotation': rotation,
            'camera_intrinsic': []
        }
        self.nuscenes_data['calibrated_sensor'].append(calibrated_sensor)
        
        channel = "LIDAR_TOP"
        filename = f"samples/{channel}/{os.path.basename(pcd_url)}"
        
        sample_data_token = ego_pose_token
        sample_data = {
            'token': sample_data_token,
            'sample_token': sample['token'],
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': calibrated_sensor_token,
            'filename': filename,
            'fileformat': pcd_url.split('.')[-1],
            'is_key_frame': True,
            'height': 0,
            'width': 0,
            'timestamp': sample['timestamp'],
            'prev': '',
            'next': ''
        }

        if channel not in self.sample_data_tokens:
            self.sample_data_tokens[channel] = []
        if len(self.sample_data_tokens[channel]) > 0:
            prev_token = self.sample_data_tokens[channel][-1]
            sample_data['prev'] = prev_token
            
            for prev_data in self.nuscenes_data['sample_data']:
                if prev_data['token'] == prev_token:
                    prev_data['next'] = sample_data_token
                    break
        
        self.sample_data_tokens[channel].append(sample_data_token)
        self.nuscenes_data['sample_data'].append(sample_data)
        
        if channel not in sample['data']:
            sample['data'][channel] = sample_data_token
        
        target_path = os.path.join(self.output_dir, filename)
        if os.path.exists(target_path):
            return
        try:
            if pcd_url.startswith(('http://', 'https://')):
                download_file(pcd_url, target_path)
            else:
                print(f"无法处理的文件路径: {pcd_url}")
        except Exception as e:
            print(f"处理文件失败: {e}")
    
    def _process_camera_data(self, sample: Dict, sensor_token: str, img_url: str, ego_pose_token: str, frame_idx: int, camera_idx: int):
        """处理相机数据"""
        
        calibrated_sensor_token = self._generate_token()
        calibrated_sensor = {
            'token': calibrated_sensor_token,
            'sensor_token': sensor_token,
            'translation': [0, 0, 0],
            'rotation': [1, 0, 0, 0],
            'camera_intrinsic': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
        self.nuscenes_data['calibrated_sensor'].append(calibrated_sensor)
        channel = self.sensor_mapping.get(camera_idx, f'CAM_{camera_idx+1}')
        filename = f"samples/{channel}/{os.path.basename(img_url)}"

        img_height, img_width = 900, 1600
        target_path = os.path.join(self.output_dir, filename)
        try:
            from PIL import Image
            with Image.open(target_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"获取图片尺寸失败: {e}")

        sample_data_token = f"{ego_pose_token}_{camera_idx}"
        sample_data = {
            'token': sample_data_token,
            'sample_token': sample['token'],
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': calibrated_sensor_token,
            'filename': filename,
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': img_height,
            'width': img_width,
            'timestamp': sample['timestamp'],
            'prev': '',
            'next': ''
        }

        if channel not in self.sample_data_tokens:
            self.sample_data_tokens[channel] = []
        
        if len(self.sample_data_tokens[channel]) > 0:
            prev_token = self.sample_data_tokens[channel][-1]
            sample_data['prev'] = prev_token
            
            for prev_data in self.nuscenes_data['sample_data']:
                if prev_data['token'] == prev_token:
                    prev_data['next'] = sample_data_token
                    break
        
        self.sample_data_tokens[channel].append(sample_data_token)
        self.nuscenes_data['sample_data'].append(sample_data)
        
        if channel not in sample['data']:
            sample['data'][channel] = sample_data_token
        
        target_path = os.path.join(self.output_dir, filename)
        if os.path.exists(target_path):
            return
        try:
            if img_url.startswith(('http://', 'https://')):
                download_file(img_url, target_path)
            else:
                print(f"无法处理的文件路径: {img_url}")
        except Exception as e:
            print(f"处理文件失败: {e}")
    
    def _process_annotations(self, sample: Dict, sequence: Dict, frame_idx: int):
        """处理标注数据"""
        if 'labels' not in sequence:
            print(f"序列中没有找到标签数据")
            return
        
        frame_labels = []
        for label_item in sequence.get('labels', []):
                
            label = label_item['data']
            if label.get('frameIndex') != frame_idx:
                continue
                
            if label.get('drawType') != 'box3d':
                continue
                
            frame_labels.append(label)
        
        if not frame_labels:
            print(f"帧 {frame_idx} 没有找到标签数据")
            return

        for label in frame_labels:
            label_id = label.get('id', '')
            label_name = label.get('label', '')
            points = label.get('points', [])
            
            x, y, z = points[0:3]
            roll, pitch, yaw = points[3:6]
            length, width, height = points[6:9]

            instance_token = self._get_or_create_instance(label_id, label_name)
            
            attribute_tokens = []
            label_attributes = label.get('attributes', {})
        
            if label_attributes:
                if 'self' in label_attributes:
                    label_attributes = label_attributes['self']
                attr_order = self._get_attribute_order_for_label(label_name)
                attr_parts = []
                for attr_key in attr_order:
                    if attr_key in label_attributes:
                        attr_parts.append(label_attributes[attr_key])
                
                if not attr_parts and label_attributes:
                    attr_parts = list(label_attributes.values())
                    
                full_attr_name = f"{label_name}." + ".".join(attr_parts)
                attr_token = self.attribute_map.get(full_attr_name)
                
                if attr_token:
                    attribute_tokens.append(attr_token)
            
            annotation_token = self._generate_token()
            annotation = {
                'token': annotation_token,
                'sample_token': sample['token'],
                'instance_token': instance_token,
                'visibility_token': self._get_visibility_token(4),
                'attribute_tokens': attribute_tokens,
                'translation': [float(x), float(y), float(z)],
                'size': [float(length), float(width), float(height)],
                'rotation': euler_to_quaternion([float(roll), float(pitch), float(yaw)]),
                'prev': '',
                'next': '',
                'num_lidar_pts': label.get('pointsInFrame', 10),
                'num_radar_pts': 0,
            }
            
            self.nuscenes_data['sample_annotation'].append(annotation)

    def _get_attribute_order_for_label(self, label_name: str) -> List[str]:
        """
        获取指定标签的属性顺序
        
        Args:
            label_name: 标签名称
            
        Returns:
            属性键的有序列表
        """
        if 'task' in self.moore_data and 'setting' in self.moore_data['task'] and 'labelConfig' in self.moore_data['task']['setting']:
            label_configs = self.moore_data['task']['setting']['labelConfig']
            
            for label_config in label_configs:
                if label_config.get('label') == label_name:
                    attr_order = []
                    for attr in label_config.get('attributes', []):
                        attr_order.append(attr.get('label', ''))
                    return attr_order
        
        return []

    def _get_or_create_instance(self, label_id: str, label_name: str) -> str:
        """获取或创建实例"""
        for instance in self.nuscenes_data['instance']:
            if instance.get('instance_id') == label_id:
                return instance['token']
        
        category_token = self._get_category_token(label_name)
        instance_token = self._generate_token()
        instance = {
            'token': instance_token,
            'category_token': category_token,
            'nbr_annotations': 1,
            'first_annotation_token': '',
            'last_annotation_token': '',
            'instance_id': label_id
        }
        
        self.nuscenes_data['instance'].append(instance)
        return instance_token

    def _get_category_token(self, label_name: str) -> str:
        """获取或创建类别"""
        for category in self.nuscenes_data['category']:
            if category['name'] == label_name:
                return category['token']
        
        category_token = self._generate_token()
        category = {
            'token': category_token,
            'name': label_name,
            'description': f"Category {label_name}"
        }
        
        self.nuscenes_data['category'].append(category)
        return category_token

    def _get_visibility_token(self, level: int) -> str:
        """获取或创建可见性级别"""
        for visibility in self.nuscenes_data['visibility']:
            if visibility['token'] == str(level):
                return visibility['token']
        
        visibility_token = str(level)
        visibility = {
            'token': str(level),
            'level': 'v80-100',
            'description': f"Visibility level {level}"
        }
        
        self.nuscenes_data['visibility'].append(visibility)
        return visibility_token

    def _update_instance_annotation_links(self):
        """更新实例和标注之间的链接"""
        instance_annotations = {}
        for annotation in self.nuscenes_data['sample_annotation']:
            instance_token = annotation['instance_token']
            if instance_token not in instance_annotations:
                instance_annotations[instance_token] = []
            instance_annotations[instance_token].append(annotation)
        
        for instance in self.nuscenes_data['instance']:
            instance_token = instance['token']
            if instance_token in instance_annotations:
                annotations = instance_annotations[instance_token]
                instance['nbr_annotations'] = len(annotations)
                
                annotations.sort(key=lambda x: self._get_sample_timestamp(x['sample_token']))
                
                instance['first_annotation_token'] = annotations[0]['token']
                instance['last_annotation_token'] = annotations[-1]['token']
                
                for i in range(len(annotations) - 1):
                    annotations[i]['next'] = annotations[i + 1]['token']
                    annotations[i + 1]['prev'] = annotations[i]['token']

    def _get_sample_timestamp(self, sample_token: str) -> int:
        """获取样本的时间戳"""
        for sample in self.nuscenes_data['sample']:
            if sample['token'] == sample_token:
                return sample['timestamp']
        return 0

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
            if 'task' in self.moore_data and 'setting' in self.moore_data['task'] and 'labelConfig' in self.moore_data['task']['setting']:
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
    
    def moore_json2nuscenes_3dod(self) -> str:
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


if __name__ == "__main__":
    input_file = r'G:\PythonProject\mooresdk\test\moore_format_3Dseg.json'
    output_dir = r'G:\PythonProject\mooresdk\test\nuscenes_test'
    lidarseg = True
    # args = parser.parse_args()
    # export_nuscenes(input_file, output_dir)
    with open(input_file, 'r', encoding='utf-8') as f:
        moore_data = json.load(f)

    converter = ExportNuscenes(moore_data, output_dir)
    converter.moore_json2nuscenes_lidarseg()
    