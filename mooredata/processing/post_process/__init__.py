#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ...factory.data_factory import PostProcessFactory


class PostProcess(object):
    postprocess_f = PostProcessFactory()

    @classmethod
    def coco_split(cls, data_path, out_pat=None, test_size=0.3, train_size=None, shuffle=True):
        cls.postprocess_f.post_process_product(data_path, out_pat).split(test_size, train_size, shuffle)

    @classmethod
    def coco_merge(cls, data_path, out_pat=None, merged_file_name=None):
        cls.postprocess_f.post_process_product(data_path, out_pat).merge(merged_file_name)
