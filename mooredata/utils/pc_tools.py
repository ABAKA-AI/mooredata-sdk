# -*-coding:utf-8 -*-
import math
import re
import struct
import cv2
import zlib
import numpy as np
import pandas as pd
from typing import Tuple, Dict


HEADER_PATTERN = re.compile(
    r'^(VERSION|FIELDS|SIZE|TYPE|COUNT|WIDTH|HEIGHT|VIEWPOINT|POINTS|DATA)'
)
TYPE_MAP = {
    ('U', 1): np.uint8, ('U', 2): np.uint16, ('U', 4): np.uint32, ('U', 8): np.uint64,
    ('I', 1): np.int8, ('I', 2): np.int16, ('I', 4): np.int32, ('I', 8): np.int64,
    ('F', 4): np.float32, ('F', 8): np.float64
}


def lzf_decompress(in_data, out_len):
    """

    :param in_data: Compressed data
    :param out_len: Length of decompressed data
    :return: Decompressed data
    """
    if not in_data:
        return bytearray(out_len)
        
    in_len = len(in_data)
    out_data = bytearray(out_len)
    in_ptr = 0
    out_ptr = 0
    
    try:
        while in_ptr < in_len and out_ptr < out_len:
            ctrl = in_data[in_ptr]
            in_ptr += 1
            
            if ctrl < 32:
                length = ctrl + 1
                if in_ptr + length > in_len or out_ptr + length > out_len:
                    print(f"警告：LZF解压缩时超出范围，in_ptr={in_ptr}, length={length}, in_len={in_len}, out_ptr={out_ptr}, out_len={out_len}")
                    length = min(length, in_len - in_ptr, out_len - out_ptr)
                
                out_data[out_ptr:out_ptr+length] = in_data[in_ptr:in_ptr+length]
                out_ptr += length
                in_ptr += length
            else:
                length = ctrl >> 5
                if length == 7:
                    if in_ptr < in_len:
                        length += in_data[in_ptr]
                        in_ptr += 1
                    else:
                        break
                
                reference = out_ptr - ((ctrl & 0x1f) << 8) - 1
                if in_ptr < in_len:
                    reference -= in_data[in_ptr]
                    in_ptr += 1
                else:
                    break
                
                if reference < 0:
                    print(f"警告：LZF解压缩时引用位置无效，reference={reference}")
                    break
                
                length += 2
                if out_ptr + length > out_len:
                    print(f"警告：LZF解压缩时超出输出范围，out_ptr={out_ptr}, length={length}, out_len={out_len}")
                    length = min(length, out_len - out_ptr)
                
                for i in range(length):
                    if reference + i < out_ptr:
                        out_data[out_ptr] = out_data[reference + i]
                        out_ptr += 1
                    else:
                        print(f"警告：LZF解压缩时引用位置超出当前位置，reference={reference}, i={i}, out_ptr={out_ptr}")
                        break
    except Exception as e:
        print(f"LZF解压缩过程中发生错误：{e}")
    
    if out_ptr < out_len:
        print(f"警告：LZF解压缩未达到预期大小，已解压{out_ptr}字节，预期{out_len}字节")
    
    return bytes(out_data)


def parse_header(pcd_path: str) -> Tuple[Dict, int]:
    """
    Parsing PCD file headers
    :param pcd_path:
    :return:
    """
    headers = {}
    data_start = 0
    encoding = 'utf-8'
    try:
        with open(pcd_path, 'r') as f:
            lines = [next(f) for _ in range(11)]
    except UnicodeDecodeError:
        with open(pcd_path, 'rb') as f:
            lines = [next(f).decode('latin-1') for _ in range(11)]
        encoding = 'binary'

    for idx, line in enumerate(lines):
        if match := HEADER_PATTERN.match(line):
            key = match.group()
            parts = line.strip().split()
            if key == 'DATA':
                headers[key] = parts[1]
                data_start = idx + 1
            else:
                headers[key] = parts[1:] if len(parts) > 1 else parts[1]

    # 类型转换关键字段
    headers['POINTS'] = int(headers.get('POINTS', 0)[0])
    headers['WIDTH'] = int(headers.get('WIDTH', [0])[0])
    headers['HEIGHT'] = int(headers.get('HEIGHT', [0])[0])
    return headers, data_start, encoding


def build_dtype(fields: list, types: list, sizes: list, counts: list) -> np.dtype:
    """
    Constructing dtype for structured arrays
    :param fields:
    :param types:
    :param sizes:
    :param counts:
    :return:
    """
    dtype = []
    for field, type_char, size, count in zip(fields, types, sizes, counts):
        np_type = TYPE_MAP.get((type_char.upper(), int(size)), None)
        if np_type is None:
            raise ValueError(f"Unsupported type: {type_char}{size}")

        count = int(count)
        if count > 1:
            dtype.extend([(f"{field}_{i}", np_type) for i in range(count)])
        else:
            dtype.append((field, np_type))
    return np.dtype(dtype)


def read_ascii_data(pcd_path: str, data_start: int, dtype: np.dtype) -> np.ndarray:
    """
    Reading point cloud data in ASCII format
    :param pcd_path:
    :param data_start:
    :param dtype:
    :return:
    """
    try:
        return np.loadtxt(pcd_path, skiprows=data_start, dtype=dtype)
    except ValueError:
        df = pd.read_csv(
            pcd_path,
            skiprows=data_start,
            sep=r'\s+',
            header=None,
            engine='python',
            dtype=dtype.str,
            on_bad_lines='warn'
        )
        return df.iloc[:, :len(dtype.names)].to_numpy().view(dtype)


def read_binary_data(pcd_path: str, data_start: int, dtype: np.dtype,
                     encoding: str) -> np.ndarray:
    with open(pcd_path, 'rb') as f:
        for _ in range(data_start):
            f.readline()
        return np.fromfile(f, dtype=dtype)


def read_compressed_data(pcd_path: str, data_start: int, dt: np.dtype,
                         width: int, height: int) -> np.ndarray:
    with open(pcd_path, 'rb') as f:
        for _ in range(data_start):
            _ = f.readline()

        compressed_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        decompressed_size = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        compressed_data = f.read(compressed_size)

        try:
            import lzf
            decompressed_data = lzf.decompress(compressed_data, decompressed_size)
        except ImportError:
            try:
                decompressed_data = lzf_decompress(compressed_data, decompressed_size)
            except Exception as e:
                try:
                    decompressed_data = zlib.decompress(compressed_data)
                except Exception as e2:
                    return np.empty(0, dtype=dt)
        
    if len(decompressed_data) != decompressed_size:
        print(f"警告：解压后数据大小({len(decompressed_data)})与预期大小({decompressed_size})不符")
    
    total_points = width * height
    pc_points_empty = np.empty(total_points, dtype=dt)

    buffer = memoryview(decompressed_data)

    for name in dt.names:
        itemsize = dt.fields[name][0].itemsize
        bytes_total = itemsize * total_points
        
        if len(buffer) < bytes_total:
            print(f"警告：buffer中数据不足，需要{bytes_total}字节，但只有{len(buffer)}字节")
            missing_bytes = bytes_total - len(buffer)
            temp_buffer = bytearray(buffer) + bytearray(missing_bytes)
            column = np.frombuffer(temp_buffer, dt.fields[name][0])
        else:
            column = np.frombuffer(buffer[:bytes_total], dt.fields[name][0])
            buffer = buffer[bytes_total:]
        
        pc_points_empty[name] = column

    return pc_points_empty


def read_pcd(pcd_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read PCD file, return point cloud data and header information
    :param pcd_path:
    :return: points data [n x m] ndarray
             headers
    """
    headers, data_start, encoding = parse_header(pcd_path)

    dtype = build_dtype(
        headers['FIELDS'],
        headers['TYPE'],
        headers['SIZE'],
        headers['COUNT']
    )

    data_format = headers['DATA']
    if data_format == 'ascii':
        points = read_ascii_data(pcd_path, data_start, dtype)
    elif data_format == 'binary':
        data = read_binary_data(pcd_path, data_start, dtype, encoding)
        # 过滤全零数据点（根据实际需求可调整）
        # points = data[~np.all(data.view(np.uint8) == 0, axis=1)]
        points = data
    elif data_format == 'binary_compressed':
        points = read_compressed_data(
            pcd_path, data_start, dtype, headers['WIDTH'], headers['HEIGHT']
        )
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    return np.array(points.view(dtype).tolist()), headers


def write_pcd(points, out_path, head=None, data_mode='binary'):
    """
    write pcd file
    :param points: 2-d np.array
    :param out_path:
    :param head: {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"] }
    :param data_mode: ascii, binary
    :return:
    """
    point_num = points.shape[0]
    if head is None:
        head = {
            "FIELDS": ["x", "y", "z", "intensity"],
            "SIZE": ["4", "4", "4", "4"],
            "TYPE": ["F", "F", "F", "F"],
            "COUNT": ["1", "1", "1", "1"]
        }
    header = f'# .PCD v0.7 - Point Cloud Data file format\n' \
             f'VERSION 0.7\n' \
             f'FIELDS {" ".join(head["FIELDS"])}\n' \
             f'SIZE {" ".join(head["SIZE"])}\n' \
             f'TYPE {" ".join(head["TYPE"])}\n' \
             f'COUNT {" ".join(head["COUNT"])}\n' \
             f'WIDTH {point_num if "WIDTH" not in head else head["WIDTH"]}\n' \
             f'HEIGHT {"1" if "HEIGHT" not in head else head["HEIGHT"]}\n' \
             f'VIEWPOINT {"0 0 0 1 0 0 0" if "VIEWPOINT" not in head else " ".join(head["VIEWPOINT"])}\n' \
             f'POINTS {point_num}\n' \
             f'DATA {data_mode}'

    width_match = re.search(r'WIDTH (\d+)', header)
    height_match = re.search(r'HEIGHT (\d+)', header)

    width = int(width_match.group(1))
    height = int(height_match.group(1))
    if width * height != point_num:
        raise Exception('Incorrect header <WIDTH> OR <HEIGHT>')

    type_map = {('U', '1'): 'B', ('U', '2'): 'H', ('U', '4'): 'I', ('U', '8'): 'Q',
                ('F', '4'): 'f', ('F', '8'): 'd',
                ('I', '1'): 'c', ('I', '2'): 'h', ('I', '4'): 'i'}

    if data_mode == 'ascii':
        handle = open(out_path, 'w')
        handle.write(header)
        for point in points:
            str_points = [str(p) for p in point]
            string = '\n' + ' '.join(str_points)
            handle.write(string)
        handle.close()
    elif data_mode == 'binary':
        with open(out_path, 'wb') as handle:
            handle.write(header.encode())
            handle.write(b'\n')

            pack_string = ''.join(
                [type_map[(type, size)] * int(count) for type, size, count in
                 zip(head['TYPE'], head['SIZE'], head['COUNT'])]
            )
            points_string = []

            for point in points:
                binary_data = [
                    struct.pack(pack, int(point[idx]) if pack not in ['f', 'd'] else (
                        np.float32(point[idx]) if pack == 'f' else np.float64(point[idx])
                    ))
                    for idx, pack in enumerate(pack_string)
                ]
                points_string.append(b''.join(binary_data))

            handle.write(b''.join(points_string))

    elif data_mode == 'binary_compressed':
        # TODO: binary_compressed
        raise 'Temporarily unable to write binary_compressed data.'
    else:
        raise 'Unknown pcd data type.'


def pcd2bin(pcd_path, out_path):
    """
    pcd convert to bin
    :param fields: pcd FIELDS
    :param pcd_folder:
    :param bin_folder:
    :return:
    """
    data, headers = read_pcd(pcd_path)
    data.tofile(out_path)


def bin2pcd(bin_path, pcd_path, head=None, data_mode='asscii'):
    """
    Convert point cloud bin format to pcd format
    :param bin_path: bin file path
    :param pcd_path: pcd file path
    :param head: {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"] }
    :return: pcd file
    """

    if head is None:
        head = {
            "FIELDS": ["x", "y", "z", "intensity"],
            "SIZE": ["4", "4", "4", "4"],
            "TYPE": ["F", "F", "F", "F"],
            "COUNT": ["1", "1", "1", "1"]
        }

    points = np.fromfile(bin_path, dtype="float32").reshape((-1, len(head['FIELDS'])))
    write_pcd(points, pcd_path, head, data_mode)


def pcd_ascii2binary(input_file, output_file):
    point_data, headers = read_pcd(input_file)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(point_data, output_file, head, data_mode='binary')

    print('Conversion complete!')


def pcd_binary2ascii(input_file, output_file):
    point_data, headers = read_pcd(input_file)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(point_data, output_file, head, data_mode='ascii')

    print('Conversion complete!')


def filter_points_in_boxes(point_cloud, boxes_list):
    """
    Given point cloud and a list of 3D boxes, remove the points inside the boxes.

    @param point_cloud: (N, 3) numpy.ndarray, N points.
    @param boxes_list: list of boxes with format [x, y, z, roll, pitch, yaw, length, width, height].
    """
    try:
        intensity_point = point_cloud[:, 3]
    except:
        intensity_point = np.zeros(len(point_cloud))
    xyz_point = point_cloud[:, :3]
    for box in boxes_list:
        box_center = np.array(box[:3])
        rot = euler_to_rotation_matrix(box[3:6])
        box_size = np.array(box[6:])

        RT = np.eye(4)
        RT[:3, :3] = rot
        RT[:3, 3] = box_center

        box_pcd_np = np.linalg.inv(RT).dot(
            np.concatenate([xyz_point, np.ones((xyz_point.shape[0], 1))], axis=-1).T).T[:, :3]
        mask = np.all((box_pcd_np >= -0.5 * box_size) & (box_pcd_np <= 0.5 * box_size), axis=1)
        xyz_point = xyz_point[~mask]
        intensity_point = intensity_point[~mask]
    filtered_point = np.concatenate([xyz_point, intensity_point.reshape(-1, 1)], axis=1)
    return filtered_point


def voxel_subsample(pcd_path, voxel_size, intensity=None, iten_idx=3, output_path='./subsampled.pcd'):
    """
    Retain points within the intensity information threshold and downsample the remaining points based on voxels
    :param pcd_path: pcd format point cloud path
    :param intensity: intensity range (list) example: [20, 200]
    :param voxel_size: voxel size
    :param output_path: Point cloud file save path
    :return:
    """
    pc_points, headers = read_pcd(pcd_path)
    # We default the first four columns of the point cloud to x, y, z, intensity
    if intensity:
        points_intensity = pc_points[(pc_points[:, iten_idx] >= intensity[0]) & (pc_points[:, iten_idx] <= intensity[1])]
        points_other = pc_points[(pc_points[:, iten_idx] < intensity[0]) | (pc_points[:, iten_idx] > intensity[1])]
        voxel_coords = np.floor(points_other[:, 0:3] / voxel_size).astype(np.int32)
        voxel_indices = np.unique(voxel_coords, axis=0, return_index=True)[1]
        downsampled_points_other = points_other[voxel_indices]
        final_points = np.concatenate((points_intensity, downsampled_points_other), axis=0)
    else:
        voxel_coords = np.floor(pc_points[:, 0:3] / voxel_size).astype(np.int32)
        voxel_indices = np.unique(voxel_coords, axis=0, return_index=True)[1]
        final_points = pc_points[voxel_indices]

    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(final_points, output_path, head, data_mode='binary')


def random_subsample(pcd_path, sampling_ratio, intensity, output_path='./subsampled.pcd'):
    """
    Retain points within the intensity information threshold and randomly downsample the remaining points
    :param pcd_path: pcd format point cloud path
    :param intensity: intensity range (list) example: [20, 200]
    :param sampling_ratio: downsampling rate
    :param output_path: Point cloud file save path
    :return:
    """
    pc_points, headers = read_pcd(pcd_path)[:, :4]
    if intensity:
        points_intensity_20_200 = pc_points[(pc_points[:, 3] >= intensity[0]) & (pc_points[:, 3] <= intensity[1])]
        points_other = pc_points[(pc_points[:, 3] < intensity[0]) | (pc_points[:, 3] > intensity[1])]
        num_points = points_other.shape[0]
        num_sampled_points = int(sampling_ratio * num_points)
        sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
        downsampled_points_other = points_other[sampled_indices]
        final_points = np.concatenate((points_intensity_20_200, downsampled_points_other), axis=0)
    else:
        num_points = pc_points.shape[0]
        num_sampled_points = int(sampling_ratio * num_points)
        sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
        final_points = pc_points[sampled_indices]

    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(final_points, output_path, head)


def pnp_compute_Rt(obj_points, img_points, intrinsic, distortion):
    """
    Solve for the rotation matrix R and the translation vector t
    :param obj_points: 2-d list [[x1, y1, z1], [x2, y2. z2], [x3, y3, z3]]
    :param img_points: 2-d list [[x1, y1], [x2, y2], [x3, y3]]
    :return: R, t
    """
    objectPoints = np.array(obj_points, dtype="double")
    imagePoints = np.array(img_points, dtype="double")

    cameraMatrix = np.array([[intrinsic[0], 0, intrinsic[2]],
                             [0, intrinsic[1], intrinsic[3]],
                             [0, 0, 1]], dtype="double")
    distCoeffs = np.array(distortion, dtype="double")

    _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

    # Print the rotation and translation vectors
    return rvec, tvec


def box_points_num(point_cloud, box_center, box_orientation, box_size):
    """
    Calculate the number of points in the 3D box
    :param point_cloud: 2d-array
    :param box_center: center point [x,y,z]
    :param box_orientation: angle of rotation [r,p,y]
    :param box_size: size [l,w,h]
    :return: number of points
    """
    cx, cy, cz = box_center
    r, p, y = box_orientation
    l, w, h = box_size

    inv_rotation = euler_to_rotation_matrix([float(r), float(p), float(y)])
    t = [cx,cy,cz]

    points_rotated = (np.dot(np.linalg.inv(inv_rotation), point_cloud[:, :3].T - np.reshape(t, (3, 1)))).T

    xmin, ymin, zmin = -l / 2, -w / 2, -h / 2
    xmax, ymax, zmax = l / 2, w / 2, h / 2
    inside_x = (points_rotated[:, 0] > xmin) & (points_rotated[:, 0] < xmax)
    inside_y = (points_rotated[:, 1] > ymin) & (points_rotated[:, 1] < ymax)
    inside_z = (points_rotated[:, 2] > zmin) & (points_rotated[:, 2] < zmax)
    inside_box = inside_x & inside_y & inside_z
    num_points = np.count_nonzero(inside_box)

    return num_points


def box_points(point_cloud, box_center, box_orientation, box_size):
    """
    Get the points in the 3D box
    :param point_cloud: 2d-array
    :param box_center: center point [x,y,z]
    :param box_orientation: angle of rotation [r,p,y]
    :param box_size: size [l,w,h]
    :return: Points in 3D box 2d-array
    """
    cx, cy, cz = box_center
    r, p, y = box_orientation
    l, w, h = box_size

    inv_rotation = euler_to_rotation_matrix([float(r), float(p), float(y)])
    t = [cx, cy, cz]
    points_rotated = (np.dot(np.linalg.inv(inv_rotation), point_cloud[:, :3].T - np.reshape(t, (3, 1)))).T

    xmin, ymin, zmin = -l / 2, -w / 2, -h / 2
    xmax, ymax, zmax = l / 2, w / 2, h / 2
    inside_x = (points_rotated[:, 0] > xmin) & (points_rotated[:, 0] < xmax)
    inside_y = (points_rotated[:, 1] > ymin) & (points_rotated[:, 1] < ymax)
    inside_z = (points_rotated[:, 2] > zmin) & (points_rotated[:, 2] < zmax)
    inside_box = inside_x & inside_y & inside_z
    box_point = point_cloud[inside_box]

    return box_point


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion
    :param R: np.array(), 3X3 rotation matrix
    :return:
    """
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1) * 2
        w = S / 4
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = S / 4
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = S / 4
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = S / 4

    return [x, y, z, w]


def rotation_matrix_to_euler(R, sequence='xyz'):
    """
    Convert a rotation matrix to Euler angles given a specified rotation sequence.

    :param R: 3x3 rotation matrix
    :param sequence: Rotation sequence as a string, e.g., 'xyz', 'zyx'
    :return: A list of Euler angles [alpha, beta, gamma] in radians
    """
    if sequence == 'xyz':  # Default sequence
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            r = math.atan2(R[2, 1], R[2, 2])
            p = math.atan2(-R[2, 0], sy)
            y = math.atan2(R[1, 0], R[0, 0])
        else:
            r = math.atan2(-R[1, 2], R[1, 1])
            p = math.atan2(-R[2, 0], sy)
            y = 0
        return [r, p, y]

    elif sequence == 'zyx':
        sy = math.sqrt(R[1, 2] * R[1, 2] + R[2, 2] * R[2, 2])
        singular = sy < 1e-6
        if not singular:
            r = -math.atan2(R[1, 2], R[2, 2])
            p = -math.atan2(-R[0, 2], sy)
            y = -math.atan2(R[0, 1], R[0, 0])
        else:
            r = -math.atan2(-R[2, 1], R[2, 2])
            p = -math.atan2(-R[0, 2], sy)
            y = 0
        return [y, p, r]

    # elif sequence == 'yxz':
    #     sy = math.sqrt(R[0, 2] * R[0, 2] + R[2, 2] * R[2, 2])
    #     singular = sy < 1e-6
    #     if not singular:
    #         r = math.atan2(R[0, 2], R[2, 2])
    #         p = math.atan2(-R[1, 2], sy)
    #         y = math.atan2(R[1, 0], R[1, 1])
    #     else:
    #         r = math.atan2(-R[2, 0], R[2, 2])
    #         p = math.atan2(-R[1, 2], sy)
    #         y = 0
    #     return [y, r, p]

    else:
        raise ValueError(f"Unsupported rotation sequence: {sequence}")


def euler_to_rotation_matrix(euler):
    """
    Convert euler angles to rotation matrix
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    roll, pitch, yaw = euler
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix
    :param q:
    :return:
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < np.finfo(float).eps:
        return np.identity(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)]
    ])


def euler_to_quaternion(euler):
    """
    Convert euler angles to quaternion
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    roll, pitch, yaw = euler
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(q_x, q_y, q_z, q_w):
    """
    Convert quaternion to euler angles
    :param x:
    :param y:
    :param z:
    :param w:
    :return:
    """

    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def boxcenter2corners(center, orientation, size):
    """
    Convert 3d box to 8 corners
    :param center: center point [x,y,z]
    :param orientation: angle of rotation [r,p,y]
    :param size: size [l,w,h]
    :return: 8 corners 2d-array
    """
    x, y, z = center
    roll, pitch, yaw = orientation
    length, width, height = size

    corners = np.array([
        [-length / 2, -width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [-length / 2, -width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [length / 2, width / 2, height / 2],
        [-length / 2, width / 2, height / 2]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))

    corners = np.dot(corners, R.T)
    corners += np.array([x, y, z])

    return corners
