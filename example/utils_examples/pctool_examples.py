# -*-coding:utf-8 -*-
import numpy as np

import moore


def test_bin2pcd():
    bin_path = '../example_files/bin'
    pcd_path = '../example_files/pcd'
    moore.bin2pcd(bin_path, pcd_path)


def test_voxel_subsample_keep_intensity():
    pcd_path = "./test.pcd"
    intensity = [20, 200]
    voxel_size = 0.3
    output_path = './test.pcd'
    moore.random_subsample_keep_intensity(pcd_path, intensity, voxel_size, output_path)


def test_pcd2bin():
    bin_path = './bin'
    pcd_path = './pcd'
    moore.pcd2bin(pcd_path, bin_path)


if __name__ == '__main__':
    # test_pcd2bin()
    bin_url = "./test.bin"
    points = np.fromfile(bin_url, dtype="float32").reshape((-1, 4))
    print(points)
