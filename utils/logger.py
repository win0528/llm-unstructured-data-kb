"""
日志系统
提供统一的日志记录功能
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "llm_preprocess",
    log_file: str = None,
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    配置并返回日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        format: 日志格式
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "llm_preprocess") -> logging.Logger:
    """获取已配置的日志记录器"""
    return logging.getLogger(name)


# 默认日志记录器
logger = setup_logger()
