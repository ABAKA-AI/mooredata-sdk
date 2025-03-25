# -*-coding:utf-8 -*-
from ..factory.data_factory import VisualFactory


class Visual(object):
    visual_f = VisualFactory()

    @classmethod
    def visual_coco(cls, source_data, data_path, out_path=None):
        cls.visual_f.visual_coco_product(source_data, data_path, out_path).visual_coco()

    @classmethod
    def visual_labelme(cls, source_data, data_path, out_path=None):
        cls.visual_f.visual_labelme_product(source_data, data_path, out_path).visual_labelme()

    @classmethod
    def visual_source(cls, source_data, out_path=None):
        cls.visual_f.visual_source_product(source_data, out_path).visual_source()

    @classmethod
    def visual_voc(cls, source_data, data_path, image_path, out_path=None):
        cls.visual_f.visual_voc_product(source_data, data_path, image_path, out_path).visual_voc()

    @classmethod
    def visual_yolo(cls, source_data, data_path, label_path, image_path, out_path=None):
        cls.visual_f.visual_yolo_product(source_data, data_path, label_path, image_path, out_path).visual_yolo()
