# import os
from datetime import datetime
# from threading import Lock
# from typing import Optional
from .get_gps import get_gps_heading

import os
from threading import Lock
from typing import Optional

def load_log_path() -> str:
    """加载日志文件路径"""
    try:
        with open('./tmp/log_path.txt', 'r', encoding='utf-8') as f:
            log_path = f.read().strip()
            print(f"Loaded log path from ./tmp/log_path.txt: {log_path}")
            return log_path
    except Exception as e:
        print(f"加载日志路径失败: {e}")
        return './logs/self_logging.log'

class MyLogger:
    _instance: Optional['MyLogger'] = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # self.log_path = log_path if log_path else './logs/self_logging.log'
        # self._ensure_log_directory()
        self._initialized = True
    
    def _ensure_log_directory(self):
        log_dir = os.path.dirname(self.log_path)
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def set_log_path(self, log_path: str):
        # save log path to ./tmp/log_path.txt

        self.log_path = log_path
        self.log_dir = os.path.dirname(self.log_path)
        self._ensure_log_directory()

    def save_log_path(self):
        with self._lock:
            try:
                with open('./tmp/log_path.txt', 'w', encoding='utf-8') as f:
                    print(f"Saving log path to ./tmp/log_path.txt: {self.log_path}")
                    f.write(self.log_path)
            except Exception as e:
                print(f"记录日志路径失败: {e}")

    def _format_message(self, message: str) -> str:
        """
        格式化日志消息
        :param level: 日志级别
        :param message: 日志消息
        :return: 格式化后的消息
        """
        gps, heading_str = get_gps_heading()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"TIME[{timestamp}] GPS[{gps}] HEADING[{heading_str}] {message}\n"
    
    def _write_log(self, message: str):
        """
        写入日志到文件（线程安全）
        :param message: 日志消息
        """
        formatted_message = self._format_message(message)
        
        with self._lock:
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_message)
            except Exception as e:
                print(f"写入日志失败: {e}")
    
    def info(self, message: str):
        """记录 INFO 级别日志"""
        self._write_log(message.replace('\n', ' '))
        # self.logger.info(message)
    
    def set_log_path(self, log_path: str):
        """
        动态修改日志文件路径
        :param log_path: 新的日志文件路径
        """
        with self._lock:
            self.log_path = log_path
            self._ensure_log_directory()

    def record_sam(self):
        # copy ./tmp/init_tracking_mask.png and ./tmp/init_tracking_view.png to self.log_dir
        print(self.log_dir)
        print(self.log_path)
        try:
            import shutil
            shutil.copy('./tmp/init_tracking_mask.png', os.path.join(self.log_dir, 'init_tracking_mask.png'))
            shutil.copy('./tmp/init_tracking_view.png', os.path.join(self.log_dir, 'init_tracking_view.png'))
        except Exception as e:
            raise RuntimeError(f"Failed to copy SAM tracking images: {e}")

    def record_tracking(self):
        tracking_image_path = './tmp/tracked_view.png'
        tracking_bbox_path = './tmp/tracked_view_bbox.json'
        # copy tracking_image_path and tracking_bbox_path to self.log_dir
        print(f'record_tracking copy {tracking_image_path} and {tracking_bbox_path} to {self.log_dir}')
        try:
            import shutil
            tracking_dir = os.path.join(self.log_dir, 'tracking_logs')
            os.makedirs(tracking_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            shutil.copy(tracking_image_path, os.path.join(tracking_dir, f'{timestamp}_' + os.path.basename(tracking_image_path)))
            shutil.copy(tracking_bbox_path, os.path.join(tracking_dir, f'{timestamp}_' + os.path.basename(tracking_bbox_path)))
        except Exception as e:
            raise RuntimeError(f"Failed to copy tracking files: {e}")

    def record_vlm_decision(self, explanation: str):
        view_path = './tmp/screenshots/current_view.png'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        decision_view_dir = os.path.join(self.log_dir, 'vlm_decision_views')
        os.makedirs(decision_view_dir, exist_ok=True)
        decision_view_path = os.path.join(decision_view_dir, f'vlm_decision_{timestamp}_view.png')
        # copy view_path to decision_view_path
        try:
            import shutil
            shutil.copy(view_path, decision_view_path)
            # write explanation to a text file
            with open(os.path.join(decision_view_dir, f'vlm_decision_{timestamp}_explanation.txt'), 'w', encoding='utf-8') as f:
                f.write(explanation)
        except Exception as e:
            raise RuntimeError(f"Failed to record VLM decision: {e}")
        # log explanation
        self.info(f"[VLM Decision] {explanation}")


# 提供全局访问函数
def get_my_logger() -> MyLogger:
    """获取日志记录器实例"""
    log_path = load_log_path()
    logger = MyLogger()
    logger.set_log_path(log_path)
    logger.info("Logger initialized.")
    return logger