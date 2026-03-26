import time
import json
import os
import sys
import builtins
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'file': kwargs.get('file', sys.stderr)})
from aircraft.msfs2024tools.flight_operations import move_forward, move_backward, move_ascend, move_descend, hover_turn_left, hover_turn_right, hover
from aircraft.safety_tools.video_depth_estimation import stop_PID_control
from aircraft.safety_tools.mode_switch_vlm import stop_PID_control_vlm
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from aircraft.utils.logger_config import setup_logger
from aircraft.utils.self_logging import get_my_logger

# Set up the logger
logger = setup_logger(__name__)


# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("pid_ctrl_operations")

class PIDController:
    """PID controller used for coordinate alignment."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Controller state variables
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None
        
    def reset(self):
        """Reset the controller state."""
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_time = None
        
    def update(self, current_pos: Tuple[float, float], 
               target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update the PID controller and compute the output.
        
        Args:
            current_pos: Current position (x, y)
            target_pos: Target position (x0, y0)
            
        Returns:
            Control output (control_x, control_y)
        """
        current_time = time.time()
        
        if self.prev_time is None:
            self.prev_time = current_time
            dt = 0.01  # Initial time interval
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.01  # Avoid division-by-zero errors
                
        # Compute the error
        error_x = target_pos[0] - current_pos[0]
        error_y = target_pos[1] - current_pos[1]
        
        # Integral term
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # Derivative term
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt
        
        # PID output
        control_x = (self.kp * error_x + 
                    self.ki * self.integral_x + 
                    self.kd * derivative_x)
        control_y = (self.kp * error_y + 
                    self.ki * self.integral_y + 
                    self.kd * derivative_y)
        
        # Update state
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
        str: Control result
    """
    logger.info("pid_tracking_control: 开始执行...")
    # return "PID control was stopped manually"
    my_logger = get_my_logger()
    my_logger.info("重要：PID开始控制")
    # yield "PID tracking control started..."
    
    max_time_s = 1.0
    tracking_bbox_data_json = "./tmp/tracked_view_bbox.json"
    # Define the sliding region
    sliding_region = {
        "x": 0.5-0.01/2,
        "y": 0.8-0.1/2,
        "width": 0.01,
        "height": 0.1
    }
    
    safety_result = stop_PID_control_vlm()
    logger.info(safety_result)
    # yield f"Safety check result: {safety_result}"
    if '否' in safety_result:
        logger.warning("安全分析结果为否，停止PID控制")
        # yield "Safety analysis returned no; stop PID control"
        return f"安全分析结果为否，停止PID控制, {safety_result}"
    
    logger.info("开始PID跟踪控制...")
    # yield "Starting PID tracking control..."
    
    # # Register signal handlers
    # signal.signal(signal.SIGUSR1, stop_pid_ctrl)  # user-defined signal 1
    # signal.signal(signal.SIGTERM, stop_pid_ctrl)  # termination signal
    
    # Create PID controllers to compute control intensity
    pid_x = PIDController(kp=2.0, ki=0.1, kd=0.5)
    pid_y = PIDController(kp=1.5, ki=0.05, kd=0.5)
    
    # Center point and boundaries of the sliding region
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
                # Read the latest tracking data
                if not os.path.exists(tracking_bbox_data_json):
                    logger.warning(f"跟踪数据文件不存在: {tracking_bbox_data_json}")
                    # yield f"Tracking data file does not exist: {tracking_bbox_data_json}"
                    time.sleep(0.1)
                    continue



                # Read the latest tracking data (cropped)
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
                # Compute the current tracking center
                tracking_center = {
                    "x": tracking_bbox_data["bbox"]["x"] + tracking_bbox_data["bbox"]["width"] / 2,
                    "y": tracking_bbox_data["bbox"]["y"] + tracking_bbox_data["bbox"]["height"] / 2
                }
                
                logger.info(f"跟踪中心: ({tracking_center['x']:.3f}, {tracking_center['y']:.3f})")
                logger.info(f"滑动区域中心: ({sliding_center_x:.3f}, {sliding_center_y:.3f})")
                # yield f"Tracking center: ({tracking_center['x']:.3f}, {tracking_center['y']:.3f}), sliding-region center: ({sliding_center_x:.3f}, {sliding_center_y:.3f})"
                
                # Check whether the target is inside the sliding region
                x_in_range = sliding_x_min <= tracking_center["x"] <= sliding_x_max
                y_in_range = sliding_y_min <= tracking_center["y"] <= sliding_y_max
                
                if x_in_range and y_in_range:
                    # Move forward when the target is inside the sliding region
                    logger.info("目标在滑动区域内，向前移动")
                    # yield "Target is inside the sliding region; moving forward"
                    move_forward(0.75)
                    
                else:
                    # Compute the PID control output
                    control_x, _ = pid_x.update(
                        (tracking_center["x"], 0), 
                        (sliding_center_x, 0)
                    )
                    _, control_y = pid_y.update(
                        (0, tracking_center["y"]), 
                        (0, sliding_center_y)
                    )
                    
                    # Control along the X axis
                    if not x_in_range:
                        x_error = tracking_center["x"] - sliding_center_x
                        control_time_x = min(abs(control_x) * 0.5, max_time_s)
                        control_time_x = max(control_time_x, 0.1)  # minimum control duration
                        
                        if tracking_center["x"] < sliding_x_min:
                            logger.info(f"X偏小，左转 {control_time_x:.2f}秒")
                            # yield f"X too small; turn left for {control_time_x:.2f} seconds"
                            hover_turn_left(control_time_x)
                            time.sleep(0.1)
                        elif tracking_center["x"] > sliding_x_max:
                            logger.info(f"X偏大，右转 {control_time_x:.2f}秒")
                            # yield f"X too large; turn right for {control_time_x:.2f} seconds"
                            hover_turn_right(control_time_x)
                            time.sleep(0.1)
                    
                    # Control along the Y axis
                    if not y_in_range:
                        y_error = tracking_center["y"] - sliding_center_y
                        control_time_y = min(abs(control_y) * 0.5, max_time_s)
                        control_time_y = max(control_time_y, 0.1)  # minimum control duration
                        
                        if tracking_center["y"] < sliding_y_min:
                            logger.info(f"Y偏小，上升 {control_time_y:.2f}秒")
                            # yield f"Y too small; ascend for {control_time_y:.2f} seconds"
                            move_ascend(control_time_y)
                        elif tracking_center["y"] > sliding_y_max:
                            logger.info(f"Y偏大，下降 {control_time_y:.2f}秒")
                            # yield f"Y too large; descend for {control_time_y:.2f} seconds"
                            move_descend(control_time_y)
                
                # Control the loop frequency
                time.sleep(0.1)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                # yield f"JSON parsing error: {e}"
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"控制循环错误: {e}")
                # yield f"Control-loop error: {e}"
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        logger.info("PID控制已停止")
        return "PID控制已停止"


# Usage example
if __name__ == "__main__":
    import logging
    logging.basicConfig(stream=sys.stderr, force=True)
    mcp.run(transport='stdio') 



