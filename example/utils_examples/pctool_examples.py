# -*-coding:utf-8 -*-
import numpy as np

import mooredata


def test_bin2pcd():
    bin_path = '../example_files/bin/1693806567200485.bin'
    pcd_path = '../example_files/pcd/1693806567200485.pcd'
    head = {
        "FIELDS": ["x", "y", "z", "intensity", "speed", " SNR"],
        "SIZE": ["4", "4", "4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1", "1", "1"]}
    mooredata.bin2pcd(bin_path, pcd_path, head=head, data_mode='binary')


def test_voxel_subsample_keep_intensity():
    pcd_path = '../example_files/pcd/1693806567200485.pcd'
    intensity = [20, 200]
    voxel_size = 0.3
    output_path = '../example_files/pcd/1693806567200485_subsample.pcd'
    mooredata.voxel_subsample(pcd_path, voxel_size=voxel_size, intensity=intensity, iten_idx=3, output_path=output_path)


def test_pcd2bin():
    bin_path = '../example_files/bin/1693806567200485.bin'
    pcd_path = '../example_files/pcd/1693806567200485.pcd'
    mooredata.pcd2bin(pcd_path, bin_path)


if __name__ == '__main__':
    # test_pcd2bin()
    test_bin2pcd()