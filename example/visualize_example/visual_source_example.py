# -*-coding:utf-8 -*-
import mooredata
from mooredata import Visual


"""
You can get your ak/sk in the platform's "Team Space" -> "Setting"
你可以在平台的"空间管理"->"空间设置"中拿到你的ak/sk
"""
ak = "Access Key"
sk = "Secret Key"
client = mooredata.Client(ak, sk)

"""
After creating an export_data task, you can see the export_data task id of the corresponding task 
in "Import/Export"->"Data Export".
创建导出任务后可以在"导入导出"->"数据导出"中看到对应任务的导出编号
"""
source_data = client.get_data('export_task_id')

out_path = './output'
# colour_dict = {
#     '车': (0, 0, 255),
#     '人': (0, 255, 0),
#     '建筑': (255, 0, 0),
# }
Visual.visual_source(source_data, out_path)