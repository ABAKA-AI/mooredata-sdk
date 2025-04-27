#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import random
from pathlib import Path

from tqdm import tqdm

from ..process import PostProcess
from mooredata.utils import load_json
from mooredata.core.moore_data import MOORE
from mooredata.io.export_data.coco.coco import COCO
from datetime import datetime


class CocoProcess(PostProcess):
    def __init__(self, data_path, out_path=None):
        super(CocoProcess, self).__init__(data_path, out_path)
        COCO.info = {
            'description': 'Convert from MOORE dataset to COCO dataset',
            'url': 'https://github.com/ABAKA-AI/mooresdk',
            'version': 'MOORE SDK V1.0',
            'year': f"{datetime.utcnow().year}",
            'contributor': '',
            'date_created': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

    def split(self, test_size=0.3, train_size=None, shuffle=True):
        def process_data(image_list, mode):
            annotations = []
            for image in tqdm(image_list):
                id = image.id
                selected_annotations = [item for item in coco_data.annotations if item['image_id'] == id]
                annotations.extend(selected_annotations)
            COCO.images = sorted(image_list, key=lambda item: item['id'])
            sorted_annotations = sorted(annotations, key=lambda item: item['image_id'])
            COCO.annotations = [{**self.moore2dict(item), 'id': idx + 1} for idx, item in
                                enumerate(sorted_annotations)]
            COCO.categories = coco_data.categories
            file_name = Path(self.data_path).parts[-1].replace('.json', f'_{mode}.json')
            self.save_labels(self.moore2dict(COCO), 'split', file_name)

        if train_size is None:
            train_size = 1 - test_size

        coco_data = MOORE(load_json(self.data_path))

        if shuffle:
            random.shuffle(coco_data.images)

        test_images = coco_data.images[:int(len(coco_data.images) * test_size)]
        train_images = coco_data.images[
                       int(len(coco_data.images) * test_size):int(len(coco_data.images) * (test_size + train_size))]
        eval_images = coco_data.images[int(len(coco_data.images) * (test_size + train_size)):]

        process_data(test_images, 'test')
        process_data(train_images, 'train')
        if len(eval_images) > 0:
            process_data(eval_images, 'eval')

    def merge(self, merged_file_name=None):
        merged_images = []
        merged_annotations = []
        merged_categories = []
        coco_paths = glob.glob(self.data_path + '/*')
        for coco_path in coco_paths:
            coco_data = MOORE(load_json(coco_path))
            image_length = len(merged_images)
            images_id = [item['id'] for item in coco_data.images]
            mapping = {id: i + 1 + image_length for i, id in enumerate(images_id)}
            merged_images.extend([{**self.moore2dict(item), 'id': mapping[item['id']]} for item in coco_data.images])
            categories = coco_data.categories

            merged_categories_dict = {item['name']: self.moore2dict(item) for item in merged_categories}
            categories_dict = {item['name']: self.moore2dict(item) for item in categories}
            for k in categories_dict:
                categories_dict[k]['source'] = "temp"
            merged_categories_dict.update(categories_dict)

            id_mapping = {}

            for i, item in enumerate(merged_categories_dict.values()):
                if item.get('source') == "temp":
                    old_id = item['id']
                    item['id'] = i
                    id_mapping[old_id] = i
            merged_categories = list(merged_categories_dict.values())
            merged_annotations.extend(
                [{**self.moore2dict(item), 'image_id': mapping[item['image_id']],
                  'category_id': id_mapping[item['category_id']]} for item in coco_data.annotations])

        for item in merged_categories:
            if 'source' in item:
                del item['source']

        sorted_merged_images = sorted(merged_images, key=lambda item: item['id'])
        sorted_merged_annotations = sorted(merged_annotations, key=lambda item: item['image_id'])
        sorted_merged_categories = sorted(merged_categories, key=lambda item: item['id'])

        COCO.images = sorted_merged_images
        COCO.annotations = sorted_merged_annotations
        COCO.categories = sorted_merged_categories

        if merged_file_name is None:
            merged_file_name = 'merged_data'
        self.save_labels(self.moore2dict(COCO), 'merge', merged_file_name + '.json')
