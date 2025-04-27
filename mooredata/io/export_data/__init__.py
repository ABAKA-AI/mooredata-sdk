# -*-coding:utf-8 -*-
from ...factory.data_factory import ExportFactory


class Export(object):
    export_f = ExportFactory()

    @classmethod
    def moore_json2coco(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_coco_product(source_data, out_path, mapping).moore_json2coco()

    @classmethod
    def sam_json2coco(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_coco_product(source_data, out_path, mapping).sam_json2coco()

    @classmethod
    def sam_json2coco_cover(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_coco_product(source_data, out_path, mapping).sam_json2coco_cover()

    @classmethod
    def moore_json2kitti(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_kitti_product(source_data, out_path, mapping).moore_json2odkitti()

    @classmethod
    def moore_json2segkitti(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_kitti_product(source_data, out_path, mapping).moore_json2segkitti()

    @classmethod
    def moore_json2labelme(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_labelme_product(source_data, out_path, mapping).moore_json2labelme()

    @classmethod
    def sam_json2labelme(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_labelme_product(source_data, out_path, mapping).sam_json2labelme()

    @classmethod
    def sam_json2labelme_cover(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_labelme_product(source_data, out_path, mapping).sam_json2labelme_cover()

    @classmethod
    def moore_json2voc(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_voc_product(source_data, out_path, mapping).moore_json2voc()

    @classmethod
    def sam_json2voc(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_voc_product(source_data, out_path, mapping).sam_json2voc()

    @classmethod
    def sam_json2voc_cover(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_voc_product(source_data, out_path, mapping).sam_json2voc_cover()

    @classmethod
    def moore_json2yolo(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_yolo_product(source_data, out_path, mapping).moore_json2yolo()

    @classmethod
    def p_mask(cls, source_data, out_path=None, mapping=None):
        cls.export_f.export_mask_product(source_data, out_path, mapping).p_mask()

    @classmethod
    def moore_json2nuscenes_lidarod(cls, source_data, out_path=None, sensor_mapping=None):
        cls.export_f.export_nuscenes_product(source_data, out_path, sensor_mapping).moore_json2nuscenes_lidarod()

    @classmethod
    def moore_json2nuscenes_lidarseg(cls, source_data, out_path=None, sensor_mapping=None):
        cls.export_f.export_nuscenes_product(source_data, out_path, sensor_mapping).moore_json2nuscenes_lidarseg()

