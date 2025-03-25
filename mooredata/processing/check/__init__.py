# -*-coding:utf-8 -*-
from ...factory.data_factory import CheckFactory


class Check(object):
    check_f = CheckFactory()

    @classmethod
    def count_labels(cls, source_data):
        count = cls.check_f.statistics_product(source_data).count_labels()
        return count

    @classmethod
    def count_aim_labels(cls, source_data, aim_label):
        count = cls.check_f.statistics_product(source_data).count_aim_labels(aim_label)
        return count

    @classmethod
    def count_drawtype(cls, source_data, draw_type):
        count = cls.check_f.statistics_product(source_data).count_drawtype(draw_type)
        return count

    @classmethod
    def count_files(cls, source_data):
        count = cls.check_f.statistics_product(source_data).count_files()
        return count

    @classmethod
    def count_images(cls, source_data):
        count = cls.check_f.statistics_product(source_data).count_images()
        return count

    @classmethod
    def unlabeld_images(cls, source_data):
        count = cls.check_f.statistics_product(source_data).unlabeld_images()
        return count

    @classmethod
    def labeled_images(cls, source_data):
        count = cls.check_f.statistics_product(source_data).labeled_images()
        return count

