# -*-coding:utf-8 -*-
import json

import moore
from moore import Export

"""
You can get your ak/sk in the platform's "Team Space" -> "Setting"
你可以在平台的"空间管理"->"空间设置"中拿到你的ak/sk
"""
ak = "eJkolB1XPXl5BecM5QCVVzigUqnvBbJxAoVvp8"
sk = "7Mv0mf9UNQfDpEZcrtTueY1F7d4AKtb0RmwXYz"
client = moore.Client(ak, sk)

"""
After creating an export task, you can see the export task id of the corresponding task 
in "Import/Export"->"Data Export".
创建导出任务后可以在"导入导出"->"数据导出"中看到对应任务的导出编号
"""
source_data = client.get_data('65fbf93a34be699061eef3d8')

# with open('test.json', 'w', encoding='utf-8') as w_f:
#     json.dump(source_data, w_f, ensure_ascii=False, indent=2)

out_path = "D:\sdktest\coco\labels"
Export.moore_json2coco(source_data=source_data, out_path=out_path)
# mapping = {"背景background": 'background', "草地lawn": 'lawn', "道路road": 'road',
#            "地形terrain": 'terrain', "障碍物obstacle": 'obstacle'}

# kitti_txt
# out_path = "./output"
# Export.moore_json2odkitti(source_data=source_data, out_path=out_path, mapping=mapping)

# kitti_label
# out_path = r"D:\sdktest/"
# Export.moore_json2segkitti(source_data=source_data, out_path=out_path)
