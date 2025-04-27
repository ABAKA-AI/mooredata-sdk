import glob
import os
import json
import numpy as np
from typing import List, Dict, Any, Union

from tqdm import tqdm


class ImportKitti:
    def __init__(self):
        self.frame_count = 0
        self.pcd_urls: List[str] = []
        self.img_urls: List[List[str]] = [[], []]
        self.locations: List[Dict[str, Any]] = []
        self.pre_data: List[Dict[str, Any]] = []
        
        # 添加相机到激光雷达的转换矩阵
        self.R = np.array([
            [0, 0, 1],    # x_lidar = z_cam
            [-1, 0, 0],   # y_lidar = -x_cam
            [0, -1, 0]    # z_lidar = -y_cam
        ])
        
    def _convert_cam_to_lidar(self, 
                             location_cam: List[float], 
                             rotation_y: float) -> tuple:
        """将相机坐标系下的位置和旋转转换到激光雷达坐标系
        Args:
            location_cam: 相机坐标系下的位置[x,y,z]
            rotation_y: 相机坐标系下的y轴旋转角
        Returns:
            tuple: (lidar位置, lidar旋转角)
        """
        # 位置转换
        location_cam = np.array(location_cam).reshape(3,1)
        location_lidar = self.R @ location_cam
        
        # 旋转角转换
        # KITTI相机系下绕y轴旋转 -> 激光雷达系下绕z轴旋转
        # 需要加上-pi/2来对齐坐标系
        rotation_lidar = rotation_y - np.pi/2
        
        return location_lidar.flatten().tolist(), rotation_lidar

    def add_frame(self, 
                  pcd_url: str,
                  kitti_label_path: str,
                  frame_id: int):
        """添加一帧数据
        Args:
            pcd_url: 点云文件URL
            kitti_label_path: KITTI标签文件路径
            frame_id: 帧ID
        """
        self.pcd_urls.append(pcd_url)
        
        # 添加位置信息
        location = {
            "name": frame_id,
            "posMatrix": [0.0] * 6  # 默认位姿
        }
        self.locations.append(location)
        
        # 读取并转换KITTI标签
        if os.path.exists(kitti_label_path):
            with open(kitti_label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                label_parts = line.strip().split(' ')
                
                # KITTI格式解析
                obj_type = label_parts[0]  # 目标类型
                truncated = float(label_parts[1])
                occluded = int(label_parts[2])
                alpha = float(label_parts[3])
                bbox = [float(x) for x in label_parts[4:8]]
                # 修改标签转换部分
                if os.path.exists(kitti_label_path):
                    with open(kitti_label_path, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        label_parts = line.strip().split(' ')
                        
                        # KITTI格式解析
                        obj_type = label_parts[0]
                        dimensions = [float(x) for x in label_parts[8:11]]  # 长宽高
                        location_cam = [float(x) for x in label_parts[11:14]]  # 相机坐标系下位置
                        rotation_y = float(label_parts[14])  # 相机坐标系下旋转角
                        
                        # 转换到激光雷达坐标系
                        location_lidar, rotation_lidar = self._convert_cam_to_lidar(
                            location_cam, rotation_y)
                        
                        # 转换为MooreData格式
                        box_data = {
                            "_id": "",
                            "id": len(self.pre_data) + 1,
                            "label": self._convert_label(obj_type),
                            "drawType": "box3d",
                            "frame": frame_id,
                            "pointsCount": None,
                            "points": [
                                location_lidar[0],  # x_lidar
                                location_lidar[1],  # y_lidar
                                location_lidar[2],  # z_lidar
                                0.0,               # roll
                                0.0,               # pitch
                                rotation_lidar,    # yaw
                                dimensions[0],     # length
                                dimensions[1],     # width
                                dimensions[2]      # height
                            ],
                            "attr": {},
                            "projection": []
                        }
                        self.pre_data.append(box_data)
        
        self.frame_count += 1
    
    def _convert_label(self, kitti_type: str) -> str:
        """转换KITTI标签类型到MooreData标签
        Args:
            kitti_type: KITTI标签类型
        Returns:
            MooreData标签类型
        """
        label_map = {
            'Car': '小汽车',
            'Van': '面包车',
            'Truck': '卡车',
            'Pedestrian': '行人',
            'Person_sitting': '坐着的人',
            'Cyclist': '骑车人',
            'Tram': '电车',
            'Misc': '其他',
            'DontCare': '未知'
        }
        return label_map.get(kitti_type, '未知')
    
    def generate_json(self) -> list[
        dict[str, Union[dict[str, Union[list[list[str]], list[str], list[dict[str, Any]]]], list[dict[str, Any]]]]]:
        """生成最终的JSON数据
        Returns:
            转换后的JSON数据
        """
        return [{
            "info": {
                "pcdUrl": self.pcd_urls,
                "imgUrls": self.img_urls,
                "locations": self.locations
            },
            "preData": self.pre_data
        }]
    
    def save_json(self, output_path: str):
        """保存JSON文件
        Args:
            output_path: 输出文件路径
        """
        json_data = self.generate_json()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

def convert_kitti_to_mooredata(
    kitti_base_path: str,
    pcd_url_list: List[str],
    output_path: str
):
    """转换KITTI数据集到MooreData格式
    Args:
        kitti_base_path: KITTI数据集根目录
        pcd_url_list: 点云文件URL列表
        output_path: 输出JSON文件路径
    """
    converter = ImportKitti()
    
    # KITTI标签目录
    label_dir = os.path.join(kitti_base_path, 'label_2')
    
    # 遍历所有帧
    for frame_id, pcd_url in tqdm(enumerate(pcd_url_list)):
        label_path = os.path.join(label_dir, f'{frame_id:06d}.txt')
        converter.add_frame(pcd_url, label_path, frame_id)
    
    # 保存结果
    converter.save_json(output_path)

if __name__ == '__main__':
    pcd_path = r'X:\development\wxj\dataset\KITTI\data_object_velodyne\training\velodyne/*.bin'
    pcd_urls = sorted(glob.glob(pcd_path))[:20]
    convert_kitti_to_mooredata(
        kitti_base_path=r"X:\development\wxj\dataset\KITTI\data_object_label_2\training",
        pcd_url_list=pcd_urls,
        output_path="G:/PythonProject/mooresdk/test/mooredata_import_23Ddynamic.json"
    )
