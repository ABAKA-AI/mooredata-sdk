# -*-coding:utf-8 -*-

import base64
import hashlib
import hmac
import json
import time
import requests
from typing import Dict, List, Any, Optional, Union
from . import exception
from ..utils import general
from ..const import MOORE_SDK_BASE_URL


class API:
    def __init__(self, AccessKey, SecretKey):
        if AccessKey == '' or AccessKey is None:
            raise exception.MooreUnauthorizedException
        elif SecretKey == '' or SecretKey is None:
            raise exception.MooreUnauthorizedException
        self.AccessKey = AccessKey
        self.SecretKey = SecretKey
    
    def signature_auth(self, timestamp: int, data: Dict[str, Any]) -> str:
        """
        通用签名方法
        :param timestamp: 时间戳
        :param data: 需要签名的数据
        :return: 签名字符串
        """
        sign_data = {'ak': self.AccessKey, 'timestamp': timestamp, **data}
        h = hmac.new(general.s2bytes(self.SecretKey), general.s2bytes(json.dumps(sign_data, separators=(',', ':'))),
                     hashlib.sha256)
        return base64.b64encode(h.digest()).decode()


class Client:
    def __init__(self, AccessKey, SecretKey):
        self.AccessKey = AccessKey
        self.SecretKey = SecretKey
        self.api = API(self.AccessKey, self.SecretKey)
    
    def _request(self, endpoint: str, data: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
        """
        通用请求方法
        :param endpoint: API端点
        :param data: 请求数据
        :param method: 请求方法，默认为POST
        :return: 响应数据
        """
        url = f"{MOORE_SDK_BASE_URL}{endpoint}"
        timestamp = int(time.time())
        
        payload = {
            "ak": self.AccessKey,
            "timestamp": timestamp,
            **data
        }
        
        headers = {
            'Authorization': self.api.signature_auth(timestamp, data),
            'Content-Type': 'application/json'
        }
        
        response = requests.request(method, url, headers=headers, data=json.dumps(payload))
        
        json_data = json.loads(response.text)
        if 'code' in json_data and json_data['code'] == 200:
            return json_data.get('data', {})
        else:
            raise exception.MooreAPIException(json_data.get('message', '未知错误'))

    def get_data(self, export_task_id: str) -> Dict[str, Any]:
        """
        获取源数据
        :param export_task_id: 导出任务ID
        :return: 数据对象
        """
        data = {"export_task_id": export_task_id}
        return self._request("/export/find-info", data)
        
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务详情
        :param task_id: 任务ID
        :return: 任务详情
        """
        data = {"task_id": task_id}
        return self._request("/task/find-info", data)
    
    def get_label_info(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        """
        获取标签列表
        :param batch_ids: 批次ID列表
        :return: 标签信息列表
        """
        data = {"batch_ids": batch_ids}
        return self._request("/task/find-labels", data)
    
    def get_item_info(self, batch_ids: List[str]) -> List[Dict[str, Any]]:
        """
        获取任务原始数据
        :param batch_ids: 批次ID列表
        :return: 原始数据列表
        """
        data = {"batch_ids": batch_ids}
        return self._request("/task/find-item-infos", data)
    
    def get_label_result(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务全部标注结果
        :param task_id: 任务ID
        :return: 标注结果
        """
        data = {"task_id": task_id}
        return self._request("/task/get-label-result", data)
    
    def get_check_result(self, task_id: str, node_id: int) -> Dict[str, Any]:
        """
        获取任务标注审核结果
        :param task_id: 任务ID
        :param node_id: 节点ID
        :return: 审核结果
        """
        data = {"task_id": task_id, "node_id": node_id}
        return self._request("/task/get-check-result", data)

    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """
        查询数据集列表
        :return: 数据集列表
        """
        return self._request("/dataset/find-list", {})

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        查询数据集详情
        :param dataset_id: 数据集ID
        :return: 数据集详情
        """
        data = {"dataset_id": dataset_id}
        return self._request("/dataset/find-info", data)
    
    