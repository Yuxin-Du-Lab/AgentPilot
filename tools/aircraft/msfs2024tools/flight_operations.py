import requests
import json
import time
import sys
import os
import builtins
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'file': kwargs.get('file', sys.stderr)})
from mcp.server.fastmcp import FastMCP
from aircraft.utils.get_gps import get_is_on_ground
from aircraft.utils.self_logging import get_my_logger

from dotenv import load_dotenv
load_dotenv()
API_URL_CTRL = os.getenv('API_URL_CTRL')
if API_URL_CTRL is None:
    raise ValueError("环境变量 API_URL_CTRL 未设置，请检查 .env 文件")
# Initialize FastMCP server
mcp = FastMCP("flight_operations")
my_logger = get_my_logger()
# API_URL = "http://10.4.147.50:5000/set"


def set_flight_parameter(param_name, param_value):
    """
    设置飞行参数
    
    Args:
        param_name (str): 参数名称
        param_value: 参数值
        
    Returns:
        str: 返回消息
    """
    # 准备请求数据
    payload = {
        "name": param_name,
        "val": param_value
    }

    print(f"正在设置参数: {param_name} = {param_value}", file=sys.stderr)

    try:
        # 确保 API_URL_CTRL 不为 None（类型检查）
        assert API_URL_CTRL is not None, "API_URL_CTRL 未正确初始化"
        # 发送PUT请求到API
        response = requests.put(API_URL_CTRL, json=payload)
        
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应
            data = response.json()
            msg = data.get('message', str(data))
            print(f"设置成功: {msg}")
            return f"设置成功: {msg}"
        else:
            if response.text:
                try:
                    error_data = response.json()
                    msg = error_data.get('message', str(error_data))
                    print(f"请求失败，状态码: {response.status_code}, 错误信息: {msg}")
                    return f"请求失败，状态码: {response.status_code}, 错误信息: {msg}"
                except:
                    print(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                    return f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}"
            else:
                print(f"请求失败，状态码: {response.status_code}")
                return f"请求失败，状态码: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return f"请求错误: {e}"
    except json.JSONDecodeError:
        print("无法解析JSON响应")
        return "无法解析JSON响应"
    except Exception as e:
        print(f"发生错误: {e}")
        return f"发生错误: {e}"

# @mcp.tool()
def move_forward(time_s):
    """
    Make the aircraft move forward
    
    Args:
        duration (float): Duration to move forward in seconds
    
    Returns:
        str: Operation result message
    """
    print(f"开始前进动作，持续时间: {time_s}秒")
    my_logger.info(f"飞行操作：开始前进动作，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION:1", 99)
    time.sleep(float(time_s))
    hover()
    print("前进动作完成")
    my_logger.info("飞行操作：前进动作完成")
    # wait for img
    time.sleep(1)
    return f"前进操作完成，持续时间: {time_s}秒"

# @mcp.tool()
def move_backward(time_s):
    """
    Make the aircraft move backward
    
    Args:
        duration (float): Duration to move backward in seconds
    
    Returns:
        None
    """
    print(f"开始后退动作，持续时间: {time_s}秒")
    my_logger.info(f"飞行操作：开始后退动作，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION:1", 0)
    time.sleep(float(time_s))
    hover()
    print("后退动作完成")
    my_logger.info("飞行操作：后退动作完成")
    pass

# @mcp.tool()
def move_left(time_s):
    """
    Make the aircraft move left
    
    Args:
        duration (float): Duration to move left in seconds
    
    Returns:
        None
    """
    print("Moving left", time_s)
    my_logger.info(f"飞行操作：开始向左移动，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("AILERON_POSITION", -1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving left done")
    my_logger.info("飞行操作：向左移动动作完成")
    pass

# @mcp.tool()
def move_right(time_s):
    """
    Make the aircraft move right
    
    Args:
        duration (float): Duration to move right in seconds
    
    Returns:
        None
    """
    print("Moving right", time_s)
    my_logger.info(f"飞行操作：开始向右移动，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("AILERON_POSITION", 1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving right done")
    my_logger.info("飞行操作：向右移动动作完成")
    pass

# @mcp.tool()
def move_ascend(time_s):
    """
    Make the aircraft move ascend
    
    Args:
        duration (float): Duration to move up in seconds
    
    """
    print("Moving up", time_s)
    my_logger.info(f"飞行操作：开始向上移动，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("ELEVATOR_POSITION", 1.0)
    time.sleep(float(time_s))
    hover()
    print("Moving up done")
    my_logger.info("飞行操作：向上移动动作完成")
    pass

@mcp.tool()
def move_descend(time_s):
    """
    Make the aircraft move descend
    
    Args:
        duration (float): Duration to move down in seconds
    
    """
    print("Moving down", time_s)
    my_logger.info(f"飞行操作：开始向下移动，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("ELEVATOR_POSITION", -1.0)
    time.sleep(min(float(time_s), 2.0))
    hover()
    print("Moving down done")
    my_logger.info("飞行操作：向下移动动作完成")
    # wait for img
    time.sleep(1)
    is_on_ground = get_is_on_ground()
    if is_on_ground:
        my_logger.info("飞机已着陆")
        return '飞机已着陆，请停止一切操作'


# @mcp.tool()
def hover():
    """
    Make the aircraft hover
    
    Args:
        None
    
    """
    print("开始悬停")
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION:1", 50)
    set_flight_parameter("RUDDER_POSITION", 0.0)
    set_flight_parameter("AILERON_POSITION", 0.0)
    set_flight_parameter("ELEVATOR_POSITION", 0.0)
    time.sleep(0.1)
    print("悬停设置完成")
    my_logger.info("飞行操作：悬停")
    pass

# @mcp.tool()
def hover_turn_left(time_s):
    """
    Make the aircraft hover and turn left
    
    Args:
        duration (float): Duration to turn left in seconds
    
    """
    print("Hover turning left", time_s)
    my_logger.info(f"飞行操作：开始悬停并向左转，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("RUDDER_POSITION", -0.05)
    time.sleep(float(time_s))
    hover()
    print("Hover turning left done")
    my_logger.info("飞行操作：悬停并向左转动作完成")
    pass

# @mcp.tool() 
def hover_turn_right(time_s):
    """
    Make the aircraft hover and turn right
    
    Args:
        duration (float): Duration to turn right in seconds
    
    """
    print("Hover turning right", time_s)
    my_logger.info(f"飞行操作：开始悬停并向右转，持续时间: {time_s}秒")
    hover()
    set_flight_parameter("RUDDER_POSITION", 0.05)
    time.sleep(float(time_s))
    hover()
    print("Hover turning right done")
    my_logger.info("飞行操作：悬停并向右转动作完成")
    pass

@mcp.tool()
def move_forward_and_descend(time_s):
    """
    Make the aircraft move forward and descend

    Args:
        duration (float): Duration to move forward and descend in seconds
    """
    print("Moving forward and descend", time_s)
    my_logger.info(f"飞行操作：开始向前移动并下降，持续时间: {time_s}秒")
    hover()
    time_s = min(float(time_s), 2.0)
    set_flight_parameter("GENERAL_ENG_THROTTLE_LEVER_POSITION:1", 99)
    time.sleep(1.0*time_s)
    hover()
    set_flight_parameter("ELEVATOR_POSITION", -1.0)
    time.sleep(0.25*time_s)
    hover()
    my_logger.info("飞行操作：向前移动并下降动作完成")
    is_on_ground = get_is_on_ground()
    if is_on_ground:
        my_logger.info("飞机已着陆")
        return '飞机已着陆，请停止一切操作'

if __name__ == "__main__":
    import logging
    logging.basicConfig(stream=sys.stderr, force=True)
    mcp.run(transport='stdio')
