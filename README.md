<div align="center">
  <img src="resources/abaka-ai.png" width="100%"/>
  <div>&nbsp;</div>
  <div align="center">
    <span><font size="150">Abaka AI website</font></span>
    <sup>
      <a href="https://www.abaka.ai/">
        <i><font size="5">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;
  </div>
  <div>&nbsp;</div>

[![PyPI - Version](https://img.shields.io/pypi/v/MooreData-SDK)](https://pypi.org/project/MooreData-SDK/)
[![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)
[![license](https://img.shields.io/github/license/mashape/apistatus)](LICENSE)

</div>

# mooredata-sdk | For Python

Welcome to `mooredata-sdk`, an open-source Software Development Kit that forms the backbone of the MooreData platform. Designed
to convert data between MooreData’s native format and widely-used, universal data structures such as COCO, YOLO, LABELME,
KITTI, VOC, mooredata-sdk helps to streamline and simplify your data operations.

The SDK is more than just a converter. It’s a swiss army knife of data processing tools. It comes loaded with an
assortment of handy utility functions often used in data processing workflows, such as Calculate the area of a polygon
or determine if a point is inside a polygon.

Whether you’re moving data, cleaning data, transforming data, or just managing it, the mooredata-sdk has got you covered
with powerful features that make your work simpler and easier. Built for developers, engineers and data scientists, this
SDK aims to make your data-heavy processes more seamless and efficient.

Stay tuned to get more details about the features, capabilities, and the simplicity mooredata-sdk brings to your data
operations.

Learn more about Abaka AI [here](https://www.abaka.ai/)!

## Overview

- [Changelog](#changelog)
- [Requirements](#requirements)
- [Installation](#installation)
- [What can we do](#what-can-we-do)
- [Usage](#usage)
- [About Us](#contact-us)
- [License](#license)

## Changelog

[2024-05-09] perf:

- Point Cloud Tools: We have optimized the code in pc tools and now support many new point cloud base operations that 
are ahead of other python third-party point cloud libraries in terms of efficiency, See [What can we do](#what-can-we-do) for available features.

[2023-11-29] perf

- Optimize mask export 

[2023-10-12] feat:

- Support for interconversion of Euler angles, quaternions and rotation matrices 

[2023-08-31] feat:

- Support pinhole camera image de-distortion and fisheye camera image de-distortion
- Support point cloud random subsampling and voxel subsampling
- Support for removing points in the 3D box of the point cloud
- Support Quaternion to Euler angle
- Support PNP <br>

[2023-07-21] mooredata-sdk v1.0.0 is released. <br>

## requirements

```sh
    python >= 3.7
    lxml
    numpy
    opencv_python
    Requests
    tqdm
    dynaconf
    scipy
    dask[dataframe]
    python-lzf
    Pillow
```

## Installation

### Install with PyPi (pip)
Please note that abava-sdk is deprecated and no longer updated, please install our sdk with the following command
```sh
pip install mooredata-sdk
```

## What can we do

### Data Format

+ [MOORE data -> COCO data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/coco/export_coco.py)
+ [MOORE data -> LABELME data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/labelme/export_labelme.py)
+ [MOORE data -> VOC data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/voc/export_voc.py)
+ [MOORE data -> YOLO data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/yolo/export_yolo.py)
+ [MOORE data -> KITTI data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/kitti/export_kitti.py)
+ [MOORE data -> MASK](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/export_format/mask/generate_mask.py)

### Data Check

+ [count labels number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L10)
+ [count specific labels number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L20)
+ [count drawtype number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L32)
+ [count file number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L45)
+ [count image number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L61)
+ [count unlabeled image number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L72)
+ [count labeled image number](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/check/statistics.py#L89)

### Data Visualization

+ [visual MOORE data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/visualize/source/visual_source.py)
+ [visual COCO data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/visualize/coco/visual_coco.py)
+ [visual LABELME data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/visualize/labelme/visual_labelme.py)
+ [visual VOC data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/visualize/voc/visual_voc.py)
+ [visual YOLO data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/visualize/yolo/visual_yolo.py)

### Computer Vision tools

+ [image data -> base64](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L15)
+ [base64 -> image data](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L29)
+ [read url image](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L41)
+ [get url image size](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L54)
+ [hexadecimal color values -> RGB](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L67)
+ [generate random RGB values](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L80)
+ [drawing boxes on the image](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L87)
+ [drawing points on the image](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L110)
+ [drawing polygons on the image](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L127)
+ [plate mode MASK -> POLYGON](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L216)
+ [MASK -> RLE](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L231)
+ [RLE -> MASK](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L244)
+ [POLYGON -> MASK](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L262)
+ [determine if the point is inside the outer rectangle](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L278)
+ [determine if the point is inside the polygon](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L334)
+ [calculate the polygon area](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L414)
+ [determining the containment of polygons](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L431)
+ [skeleton polygon](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L559)
+ [Determining Polygon Orientation](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L604)
+ [image de-distortion](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/cv_tools.py#L636)

### Point Cloud tools

+ [read PCD format point clouds](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L13)
+ [write PCD format point clouds](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L112)
+ [PCD -> BIN](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L180)
+ [BIN -> PCD](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L192)
+ [ascii -> binary](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L217)
+ [binary -> ascii](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L230)
+ [removing points from the point cloud 3D box](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L243)
+ [voxel subsampling](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L274)
+ [random subsampling](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L306)
+ [the pnp method computes rotation matrices and translation vectors](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L339)
+ [calculate the number of points in the 3D box](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L360)
+ [rotation matrix -> quaternion](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L389)
+ [rotation matrix -> euler](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L403)
+ [euler -> rotation matrix](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L416)
+ [quaternion -> rotation matrix](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L442)
+ [euler -> quaternion](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L472)
+ [quaternion -> euler](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L487)
+ [3Dbox -> corner points](https://github.com/AbakaAI/mooredata-sdk/blob/main/mooredata/utils/pc_tools.py#L512)


## Usage

### get source data

```python
import mooredata

"""
You can get your ak/sk in the platform's "Team Space" -> "Setting"
"""
ak = Access Key
sk = Secret Key
client = mooredata.Client(ak, sk)

"""
After creating an export_data task, you can see the export_data task id of the corresponding task
in "Import/Export"->"Data Export".
"""
source_data = client.get_data(export_task_id)
```

### data format

```python
from mooredata import Export

mapping = {"背景background": 'background', "草地lawn": 'lawn', "道路road": 'road'}
# coco
out_path = "./output"
Export.moore_json2coco(source_data=source_data, out_path=out_path)

```

### visualize

```python
from mooredata import Visual

data_path = "./data.json"
out_path = "./output"
Visual.visual_coco(source_data, data_path, out_path)
```

### utils

```python
import mooredata


def test_isin_external_rectangle():
    point = [55, 100]
    vertex_lst = [[50, 50], [200, 200], [200, 50], [50, 50]]
    tag = mooredata.isin_external_rectangle(point, vertex_lst)
    return tag


def test_to_string():
    test_string = b'example_string'
    string = mooredata.b2string(test_string)
    print(string)


def test_pcd2bin():
    bin_path = './bin'
    pcd_path = './pcd'
    mooredata.pcd2bin(pcd_path, bin_path)
```

Please refer to [examples.md](example/examples.md) to learn more usage about mooredata-sdk.


## About Us
Abaka AI, provides MooreData Platform (AI-Based Augmented Virtual Associate) and ACE Service (Accurate & Efficient), is committed to becoming a data navigator in the AI industry. Click to [Try Our MooreData Platform](https://app.abaka.ai/saas) .

Abaka AI collaborates with over 1000 leading global tech firms and research institutions across the Automobile AI, Generative AI, and Embodied AI sectors. Our Global Offices in Singapore, Paris, and Silicon Valley spearhead our worldwide expansion. 

Find us on [X](https://x.com/abaka_ai) | [LinkedIn](https://www.linkedin.com/company/AbakaAI) | [YouTube](https://www.youtube.com/@abaka_ai) . 

## License

mooredata-sdk is released under the MIT license.
