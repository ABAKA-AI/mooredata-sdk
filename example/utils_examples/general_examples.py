# -*-coding:utf-8 -*-
import mooredata


def test_str2dict():
    dict_string = '{"k1": "v1", "k2": "v2", "k3": "v3"}'
    test_dict = mooredata.str2dict(dict_string)
    print(test_dict)
    print(type(test_dict))


def test_getkey():
    test_dict = {'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}
    key = mooredata.get_key(test_dict, 'v1')
    print(key)


def test_chunks():
    points = [1,2,3,4,5,6,7,8,9,10,11,12]
    test_points1 = mooredata.chunks(points, 2)
    test_points2 = mooredata.chunks(points, 3)
    print(test_points1)
    print(test_points2)


def test_find_nth():
    test_string = 'aaabcdefghiiijk'
    index = mooredata.find_nth(test_string, 'l', 2)
    print(index)


def test_find_last():
    test_string = 'aaabcdefghiiijk'
    index = mooredata.find_last(test_string, 'l')
    print(index)


def test_to_bytes():
    test_string = 'aaabcdefghiiijk'
    byte = mooredata.s2bytes(test_string)
    print(byte)


def test_to_string():
    test_string = b'aaabcdefghiiijk'
    string = mooredata.b2string(test_string)
    print(string)


def test_cal_distance():
    point_a = [12, 45]
    point_b = [104, 84]
    dis = mooredata.cal_distance(point_a, point_b)
    print(dis)


def test_cal_angle():
    point_a = [0, 0]
    point_b = [-2, -2]
    angle = mooredata.cal_angle(point_a, point_b)
    print(angle)


if __name__ == '__main__':
    test_chunks()
