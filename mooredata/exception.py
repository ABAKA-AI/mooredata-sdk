# -*-coding:utf-8 -*-


class MooreException(Exception):
    code = None

    def __init__(self, message, errcode=None):
        self.message = message

        if errcode:
            self.code = errcode

        if self.code:
            super().__init__(f'<Response [{self.code}]> {message}')
        else:
            super().__init__(f'<Response> {message}')


class MooreParameterException(MooreException):
    "函数参数缺失或有误"
    code = 400


class MooreUnauthorizedException(MooreException):
    "认证错误"
    code = 401


class MooreInternetException(MooreException):
    "网络服务异常"
    code = 403


class MooreNoResourceException(MooreException):
    "无资源"
    code = 404


class MooreDrawTypeException(MooreException):
    "drawType错误"
    code = 405


class MooreNotImplementException(MooreException):
    "未实现"
    code = 406


class MooreUnknownException(MooreException):
    "未知错误"
    code = 407