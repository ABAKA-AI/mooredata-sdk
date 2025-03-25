# -*-coding:utf-8 -*-


class MooreException(Exception):
    """Moore SDK 基础异常类"""
    def __init__(self, message="Moore SDK 异常"):
        self.message = message
        super().__init__(self.message)


class MooreAPIException(MooreException):
    """API 请求异常"""
    def __init__(self, message="API 请求失败"):
        super().__init__(message)
        
        
class MooreUnauthorizedException(MooreException):
    """未授权异常"""
    def __init__(self, message="未提供有效的 AccessKey 或 SecretKey"):
        super().__init__(message)


class MooreParameterException(MooreException):
    """参数异常"""
    def __init__(self, message="参数错误"):
        super().__init__(message)
        
        
class MooreResourceNotFoundException(MooreException):
    """资源不存在异常"""
    def __init__(self, message="请求的资源不存在"):
        super().__init__(message)


class MooreNetworkException(MooreException):
    """网络异常"""
    def __init__(self, message="网络连接异常"):
        super().__init__(message)
        