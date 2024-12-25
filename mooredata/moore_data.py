# -*-coding:utf-8 -*-
import json


class MOORE:
    def __init__(self, d=None):
        if isinstance(d, dict):
            for key, value in d.items():
                self.set_attr(key, value)
        elif isinstance(d, list):
            setattr(self, 'values', [MOORE(value) for value in d])

    def __repr__(self):
        return 'molar_ai_format'

    def set_attr(self, k, v):
        if isinstance(v, dict):
            setattr(self, k, MOORE(v))
        elif isinstance(v, list):
            for value in v:
                if isinstance(value, dict):
                    setattr(self, k, [MOORE(value) for value in v])
                    break
                else:
                    setattr(self, k, v)
                    break
            if not v:
                setattr(self, k, [])
        elif isinstance(k, int):
            raise Exception("Cannot use data of type int as the key of a dictionary")
        else:
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def moore2dict(data):
    """
    Converting MOORE types to dict
    :param data:
    :return:
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                data[f'{k}'] = [moore2dict(d) for d in v]
            elif repr(v) == 'molar_ai_format':
                data[f'{k}'] = moore2dict(v)
    elif repr(data) == 'molar_ai_format':
        data = moore2dict(vars(data))
    return data


def gen_structure_json(data):
    """
    Writing dict to json files
    :param data:
    :return:
    """
    if isinstance(data, dict):
        with open('molar_ai_format.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    else:
        raise Exception('formatting error')
