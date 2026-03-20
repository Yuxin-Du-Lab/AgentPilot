# Contributors: Yuxin Du

import logging
import sys

def setup_logger(name):
    """
    设置统一的日志配置
    
    Args:
        name: 日志记录器的名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 配置日志系统
    logger = logging.getLogger(name)
    
    # 如果logger已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(console_handler)
    
    return logger 