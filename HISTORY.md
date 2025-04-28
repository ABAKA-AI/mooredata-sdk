# mooredata_sdk v1.5.0 Release Notes
## feat
1. feat: mooredata.nuscenes2mooredata_det()  nuscenes format to dynamic frames task import json  [example](example/format_examples/nuscenes2mooredata_example.py)
2. feat: mooredata.nuscenes2mooredata_lane()  nuscenes format to 4D task import json  [example](example/format_examples/nuscenes2mooredata_example.py)
3. feat：mooredata.moore_json2nuscenes_lidarod()  nuscenes format export adapted to 23D tasks [example](example/format_examples/format_nuscenes_example.py)


# 2025-04-27 mooredata_sdk v1.4.5 Release Notes
## fix
1. fix: mooredata.moore_json2nuscenes_lidarod()  Partial token association exception
## refactor
1. refactor: nuscenes format export


# 2025-03-25 mooredata_sdk v1.4.0 Release Notes
## feat
1. feat: mooredata.moore_json2nuscenes_3dod()  3D OD data format mooredata -> nuscenes  
2. feat: mooredata.moore_json2nuscenes_lidarseg()  3D seg data format mooredata -> nuscenes
3. feat: mooredata.Client.get_task_info()  Get task details
4. feat: mooredata.Client.get_label_info()  Get the list of labels
5. feat: mooredata.Client.get_item_info()  Getting the raw task data
6. feat: mooredata.Client.get_label_result()  Get all the labeling results of the task
7. feat: mooredata.Client.get_check_result()  Get the results of the task annotation review
8. feat: mooredata.Client.get_dataset_list()  Query dataset list
9. feat: mooredata.client.get_dataset_info()  Query dataset details

# 2025-02-25 mooredata_sdk v1.3.5 Release Notes
## fix
1. fix: mooredata.box_points_num()  Delete Test Code

# 2025-02-24 mooredata_sdk v1.3.4 Release Notes
## fix
1. fix: mooredata.read_url_image()  can't request url with Chinese characters
2. fix: ExportData.match_label_data()  MOORE objects can't determine attributes
## feat
1. feat: mooredata.read_pcd()  can read binary_compressed format pcd point cloud
## refactor
1. refactor: mooredata.read_pcd()

# mooredata_sdk v1.3.3 Relase Notes
## refactor
1. refactor: mooredata.boxcenter2corners()  modify the input parameter
2. refactor: mooredata.box_points()  modify the input parameter
3. refactor: mooredata.box_points_num()  modify the input parameter

# mooredata_sdk v1.3.1 Release Notes
## perf
1. perf: mooredata.rotation_matrix_to_quaternion()  Calculating quaternions based on the principal directions of rotation


# mooredata_sdk v1.0.0 Release Notes

See [What can we do](README.md#what-can-we-do)

