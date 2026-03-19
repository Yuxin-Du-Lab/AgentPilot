# Contributors: Yunqi Zhao, Yuxin Du, Qiwei Wu

import time
import json
import os
import sys
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from aircraft.msfs2024tools.flight_operations import move_forward, move_backward, move_ascend, move_descend, hover_turn_left, hover_turn_right, hover
from aircraft.safety_tools.video_depth_estimation import stop_PID_control
from aircraft.safety_tools.mode_switch_vlm import stop_PID_control_vlm
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from aircraft.utils.logger_config import setup_logger
from aircraft.utils.self_logging import get_my_logger
0
# 设置日志记录器
logger = setup_logger(__name__)


# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("pid_ctrl_operations")

class PIDController:
    """PID控制器类，用于坐标对准控制"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        """
        初始化PID控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益  
            kd: 微分增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 控制状态变量
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None
        
    def reset(self):
        """重置控制器状态"""
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None
        
    def update(self, current_pos: Tuple[float, float], 
               target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        更新PID控制器并计算输出
        
        Args:
            current_pos: 当前位置 (x, y)
            target_pos: 目标位置 (x0, y0)
            
        Returns:
            控制输出 (control_x, control_y)
        """
        current_time = time.time()
        
        if self.prev_time is None:
            self.prev_time = current_time
            dt = 0.01  # 初始时间间隔
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.01  # 避免除零错误
                
        # 计算误差
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        
        # 积分项
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # 微分项
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt
        
        # PID输出
        control_x = (self.kp * error_x + 
                    self.ki * self.integral_x + 
                    self.kd * derivative_x)
        control_y = (self.kp * error_y + 
                    self.ki * self.integral_y + 
                    self.kd * derivative_y)
        
        # 更新状态
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        return control_x, control_y

@mcp.tool()
def pid_tracking_control():
    """
    This tool is used to control the aircraft to track the sliding region.
    
    Args:
        None
    Returns:
        str: 控制结果
    """
    logger.info("pid_tracking_control: 开始执行...")
    # return "PID控制已手动停止"
    my_logger = get_my_logger()
    my_logger.info("重要：PID开始控制")
    # yield "PID跟踪控制开始执行..."
    
    max_time_s = 1.0
    tracking_bbox_data_json = "./tmp/tracked_view_bbox.json"
    # 定义滑动区域
    sliding_region = {
        "x": 0.5-0.01/2,
        "y": 0.8-0.1/2,
        "width": 0.01,
        "height": 0.1
    }
    
    safety_result = stop_PID_control_vlm()
    logger.info(safety_result)
    # yield f"安全检查结果: {safety_result}"
    if '否' in safety_result:
        logger.warning("安全分析结果为否，停止PID控制")
        # yield "安全分析结果为否，停止PID控制"
        return f"安全分析结果为否，停止PID控制, {safety_result}"
    
    logger.info("开始PID跟踪控制...")
    # yield "开始PID跟踪控制..."
    
    # # 注册信号处理器
    # signal.signal(signal.SIGUSR1, stop_pid_ctrl)  # 用户自定义信号1
    # signal.signal(signal.SIGTERM, stop_pid_ctrl)  # 终止信号
    
    # 创建PID控制器用于计算控制强度
    pid_x = PIDController(kp=2.0, ki=0.1, kd=0.5)
    pid_y = PIDController(kp=1.5, ki=0.05, kd=0.5)
    
    # 滑动区域的中心点和边界
    sliding_center_x = sliding_region["x"] + sliding_region["width"] / 2
    sliding_center_y = sliding_region["y"] + sliding_region["height"] / 2
    sliding_x_min = sliding_region["x"]
    sliding_x_max = sliding_region["x"] + sliding_region["width"]
    sliding_y_min = sliding_region["y"]
    sliding_y_max = sliding_region["y"] + sliding_region["height"]
    
    try:
        pid_cnt = 0
        while True:
            try:
                with open("./tmp/control_signal.txt", "w", encoding="utf-8") as f:
                    f.write("PID has control")
                # 读取最新的跟踪数据
                if not os.path.exists(tracking_bbox_data_json):
                    logger.warning(f"跟踪数据文件不存在: {tracking_bbox_data_json}")
                    # yield f"跟踪数据文件不存在: {tracking_bbox_data_json}"
                    time.sleep(0.1)
                    continue



                # 读取最新的跟踪数据（cropped）
                if pid_cnt % 5 == 0:
                    safety_result = stop_PID_control_vlm()
                    logger.info(safety_result)
                    if '否' in safety_result:
                        logger.warning("安全分析结果为否，停止PID控制")
                        my_logger.info("重要：PID停止控制，进入VLM智能体控制")
                        return f"安全分析结果为否，停止PID控制, {safety_result}"
                pid_cnt += 1

                with open(tracking_bbox_data_json, 'r', encoding='utf-8') as f:
                    tracking_bbox_data = json.load(f)
                
                if tracking_bbox_data["frame"] == -1:
                    logger.warning("tracking中断，安全分析结果为否，json fail, 停止PID控制")
                    my_logger.info("重要：PID停止控制，进入VLM智能体控制")
                    return f"tracking中断，安全分析结果为否，json fail, 停止PID控制"
                my_logger.record_tracking()
                # 计算当前跟踪中心
                tracking_center = {
                    "x": tracking_bbox_data["bbox"]["x"] + tracking_bbox_data["bbox"]["width"] / 2,
                    "y": tracking_bbox_data["bbox"]["y"] + tracking_bbox_data["bbox"]["height"] / 2
                }
                
                logger.info(f"跟踪中心: ({tracking_center['x']:.3f}, {tracking_center['y']:.3f})")
                logger.info(f"滑动区域中心: ({sliding_center_x:.3f}, {sliding_center_y:.3f})")
                # yield f"跟踪中心: ({tracking_center['x']:.3f}, {tracking_center['y']:.3f}), 滑动区域中心: ({sliding_center_x:.3f}, {sliding_center_y:.3f})"
                
                # 判断是否在滑动区域内
                x_in_range = sliding_x_min <= tracking_center["x"] <= sliding_x_max
                y_in_range = sliding_y_min <= tracking_center["y"] <= sliding_y_max
                
                if x_in_range and y_in_range:
                    # 在滑动区域内，向前移动
                    logger.info("目标在滑动区域内，向前移动")
                    # yield "目标在滑动区域内，向前移动"
                    move_forward(0.75)
                    
                else:
                    # 计算PID控制输出
                    control_x, _ = pid_x.update(
                        (tracking_center["x"], 0), 
                        (sliding_center_x, 0)
                    )
                    _, control_y = pid_y.update(
                        (0, tracking_center["y"]), 
                        (0, sliding_center_y)
                    )
                    
                    # X方向控制
                    if not x_in_range:
                        x_error = tracking_center["x"] - sliding_center_x
                        control_time_x = min(abs(control_x) * 0.5, max_time_s)
                        control_time_x = max(control_time_x, 0.1)  # 最小控制时间
                        
                        if tracking_center["x"] < sliding_x_min:
                            logger.info(f"X偏小，左转 {control_time_x:.2f}秒")
                            # yield f"X偏小，左转 {control_time_x:.2f}秒"
                            hover_turn_left(control_time_x)
                            time.sleep(0.1)
                        elif tracking_center["x"] > sliding_x_max:
                            logger.info(f"X偏大，右转 {control_time_x:.2f}秒")
                            # yield f"X偏大，右转 {control_time_x:.2f}秒"
                            hover_turn_right(control_time_x)
                            time.sleep(0.1)
                    
                    # Y方向控制
                    if not y_in_range:
                        y_error = tracking_center["y"] - sliding_center_y
                        control_time_y = min(abs(control_y) * 0.5, max_time_s)
                        control_time_y = max(control_time_y, 0.1)  # 最小控制时间
                        
                        if tracking_center["y"] < sliding_y_min:
                            logger.info(f"Y偏小，上升 {control_time_y:.2f}秒")
                            # yield f"Y偏小，上升 {control_time_y:.2f}秒"
                            move_ascend(control_time_y)
                        elif tracking_center["y"] > sliding_y_max:
                            logger.info(f"Y偏大，下降 {control_time_y:.2f}秒")
                            # yield f"Y偏大，下降 {control_time_y:.2f}秒"
                            move_descend(control_time_y)
                
                # 控制循环频率
                time.sleep(0.1)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                # yield f"JSON解析错误: {e}"
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"控制循环错误: {e}")
                # yield f"控制循环错误: {e}"
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        logger.info("PID控制已停止")
        return "PID控制已停止"


# 使用示例
if __name__ == "__main__":
    mcp.run(transport='stdio') 





