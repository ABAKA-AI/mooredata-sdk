#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import abstractmethod


class DataFactory():
    """
    abstract factory
    """
    type = ""

    @abstractmethod
    def export_coco_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_kitti_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_labelme_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_voc_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_yolo_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_mask_product(self, source_data, out_path=None, mapping=None):
        pass

    @abstractmethod
    def export_nuscenes_product(self, source_data, out_path=None, sensor_mapping=None):
        pass

    @abstractmethod
    def import_nuscenes_product(self, nuscenes_root, output_dir, oss_root, predata=False):
        pass
    
    @abstractmethod
    def visual_coco_product(self, source_data, data_path, out_path=None):
        pass

    @abstractmethod
    def visual_labelme_product(self, source_data, data_path, out_path=None):
        pass

    @abstractmethod
    def visual_source_product(self, source_data, out_path=None):
        pass

    @abstractmethod
    def visual_voc_product(self, source_data, data_path, image_path, out_path=None):
        pass

    @abstractmethod
    def visual_yolo_product(self, source_data, data_path, label_path, image_path, out_path=None):
        pass

    @abstractmethod
    def statistics_product(self, source_data):
        pass

    @abstractmethod
    def postprocess_coco_prodect(self, data_path, out_path=None):
        pass
    

class ExportFactory(DataFactory):
    def __init__(self):
        self.type = "EXPORT"

    def export_coco_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.coco.export_coco import ExportCoco
        return ExportCoco(source_data, out_path, mapping)

    def export_kitti_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.kitti.export_kitti import ExportKitti
        return ExportKitti(source_data, out_path, mapping)

    def export_labelme_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.labelme.export_labelme import ExportLabelme
        return ExportLabelme(source_data, out_path, mapping)

    def export_voc_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.voc.export_voc import ExportVoc
        return ExportVoc(source_data, out_path, mapping)

    def export_yolo_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.yolo.export_yolo import ExportYolo
        return ExportYolo(source_data, out_path, mapping)

    def export_mask_product(self, source_data, out_path=None, mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.mask.generate_mask import ExportMask
        return ExportMask(source_data, out_path, mapping)

    def export_nuscenes_product(self, source_data, out_path=None, sensor_mapping=None):
        print(self.type, "process has been created.")
        from ..io.export_data.nuscenes.export_nuscenes import ExportNuscenes
        return ExportNuscenes(source_data, out_path, sensor_mapping)


class ImportFactory(DataFactory):
    def __init__(self):
        self.type = "IMPORT"

    def import_nuscenes_product(self, nuscenes_root, output_json_path, oss_root, predata=False, json_file_name=None, lidar_name=None, bin_col=4, intensity_idx=3):
        print(self.type, "process has been created.")
        from ..io.import_data.nuscenes.import_nuscenes import ImportNuscenes
        return ImportNuscenes(nuscenes_root, output_json_path, oss_root, predata, json_file_name, lidar_name, bin_col, intensity_idx)


class VisualFactory(DataFactory):
    def __init__(self):
        self.type = "VISUAL"

    def visual_coco_product(self, source_data, data_path, out_path=None):
        from ..visualization.coco.visual_coco import VisualCoco
        print(self.type, "process has been created.")
        return VisualCoco(source_data, data_path, out_path)

    def visual_labelme_product(self, source_data, data_path, out_path=None):
        from ..visualization.labelme.visual_labelme import VisualLabelme
        print(self.type, "process has been created.")
        return VisualLabelme(source_data, data_path, out_path)

    def visual_source_product(self, source_data, out_path=None):
        from ..visualization.source.visual_source import VisualSource
        print(self.type, "process has been created.")
        return VisualSource(source_data, out_path)

    def visual_voc_product(self, source_data, data_path, image_path, out_path=None):
        from ..visualization.voc.visual_voc import VisualVoc
        print(self.type, "process has been created.")
        return VisualVoc(source_data, data_path, image_path, out_path)

    def visual_yolo_product(self, source_data, data_path, label_path, image_path, out_path=None):
        from ..visualization.yolo.visual_yolo import VisualYolo
        print(self.type, "process has been created.")
        return VisualYolo(source_data, data_path, label_path, image_path, out_path)


class CheckFactory(DataFactory):
    def __init__(self):
        self.type = "CHECK"

    def statistics_product(self, source_data):
        from ..processing.check.statistics import Statistic
        print(self.type, "process has been created.")
        return Statistic(source_data)


class PostProcessFactory(DataFactory):
    def __init__(self):
        self.type = "PostProcess"

    def post_process_product(self, data_path, out_path=None):
        from ..processing.post_process.coco.coco_postprocess import CocoProcess
        print(self.type, "post-process has been created.")
        return CocoProcess(data_path, out_path)

