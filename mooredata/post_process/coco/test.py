#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time         : 2024/3/26 17:40
# @Author       : Wu Xinjun
# @Site         : 
# @File         : test.py
# @Project      : opensource
# @Software     : PyCharm
# @Description  : 
"""
a = [
{
      "supercategory": "",
      "id": 0,
      "name": "MASK(split)"
    },
    {
      "supercategory": "",
      "id": 1,
      "name": "MASK(merge)"
    }
]

b = [
{
      "supercategory": "",
      "id": 0,
      "name": "MASK(split)"
    },
{
      "supercategory": "",
      "id": 1,
      "name": "rec"
    },
    {
      "supercategory": "",
      "id": 2,
      "name": "polyg"
    }
]

# a_names = {item['name'] for item in a}
# for b_item in b:
#     if b_item['name'] not in a_names:
#         b_item['id'] += len(a)
#
# a_names = {item['name'] for item in a}
# b = [b_item for b_item in b if b_item['name'] not in a_names]

# a_names = {item['name']: item for item in a}
# for b_item in b[:]:  # 现在我们遍历一个b的切片副本
#     if b_item['name'] not in a_names:
#         b_item['id'] += len(a)
#     else:
#         b.remove(b_item)  # 如果name在a中，我们从b中移除它
#
# c = a + b

# a_names = {item['name']: item for item in a}
# b_unique = []  # 创建一个新的 b 列表
# for b_item in b:
#     if b_item['name'] in a_names:
#         continue # 如果 name 在 a 中，跳过该元素
#     b_item['id'] += len(a)
#     b_unique.append(b_item)  # 添加该元素到新的 b 列表
#
# c = a + b_unique

# 创建一个字典, name字段作为键，整个字典作为值。
# dict_a = {item['name']: item for item in a}
# dict_b = {item['name']: item for item in b}
# dict_a.update(dict_b)
# # 更新b中元素的id字段。
# for i, item in enumerate(dict_a.values()):
#     item['id'] = i
# # 将结果转换为列表。
# c2 = list(dict_a.values())

# 创建一个字典, name字段作为键，整个字典作为值。
# dict_a = {item['name']: item for item in a}
# dict_b = {item['name']: item for item in b}
#
# for k in dict_b:
#     dict_b[k]['source'] = "b"  # 添加属性 'source'，只有列表 'b' 的元素需要
#
# dict_a.update(dict_b)
#
# # 创建一个字典来存储原始和新的id。
# id_mapping = {}
#
# # 更新b中元素的id字段。
# for i, item in enumerate(dict_a.values()):
#     if item.get('source') == "b":  # 判断元素是否来自'b'
#         old_id = item['id']
#         item['id'] = i
#         id_mapping[old_id] = i   # 记录原始和新的id
#
# # 将结果转换为列表。
# c = list(dict_a.values())

# 创建两个单独的字典，一个用于储存数据，一个用于映射名称到索引
# name_dict = {item['name']: index for index, item in enumerate(a)}
# data_dict = {index: item for index, item in enumerate(a)}
#
# # 对于列表b中的每一项，如果它在列表a中不存在，那么我们需要添加它
# for item in b:
#     name = item['name']
#     if name not in name_dict:
#         # 增加一个新的索引
#         new_index = len(data_dict)
#         name_dict[name] = new_index
#         # 把旧的id保存在一个新的字段之中，然后给新的元素分配一个id
#         item['old_id'] = item['id']
#         item['id'] = new_index
#         # 把新的元素添加到数据字典中去
#         data_dict[new_index] = item
#
# # 转化为列表
# new_list = [item for index, item in sorted(data_dict.items())]
for item in dict_a.values():
    if 'source' in item:
        del item['source']
print(new_list)
print(data_dict)
# print(c2)
