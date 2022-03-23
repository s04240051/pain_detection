import logging
import os
from .checkpoint import get_checkpoints_path

def get_logger(name):

    return logging.getLogger(name)


def setup_logging(log_path, checkpoint_file):

    # 获取logger对象,取名
    logger = logging.getLogger()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    log_path = get_checkpoints_path(log_path, checkpoint_file)
    log_path = os.path.join(log_path, "stdout.log")
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)