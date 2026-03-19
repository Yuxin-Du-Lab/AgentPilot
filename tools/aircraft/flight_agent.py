# Contributors: Yunqi Zhao, Yuxin Du, Chenglin Liu, Qiwei Wu,  Yuanlin Chu

# python flight_agent.py --query "fly"

# import os
# import sys
# import re

# Add the project root directory to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../..'))
# sys.path.append(project_root)

# Import the FractFlow ToolTemplate
from tools.aircraft.utils.self_logging import get_my_logger, MyLogger
# init logger
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = MyLogger()
logger.set_log_path(log_path=f'./tmp/logs/flight_agent_{timestamp}/flight_agent.log')
logger.save_log_path()
logger.info("FlightBrain_Agent started.")
logger.record_sam()

from FractFlow.tool_template import ToolTemplate
class FlightBrain_Agent(ToolTemplate):
    """Intelligent Flight Brain Agent that integrates flight control, visual analysis, and safety assessment"""
    
    SYSTEM_PROMPT = """
你是一个智能飞行大脑，负责为 Joby S4 eVTOL 调用和执行飞行决策。你的任务是使用飞行决策工具pid_ctrl和landing_ctrl，把飞机降落到直升机停机坪上。


# 工作流程
1. 首先，调用pid_ctrl，控制飞机沿下滑道下降靠近停机坪。
2. PID跟踪控制已退出后，调用landing_ctrl工具的获得飞行决策，再调用相应的飞行操作工具，把飞机降落到直升机停机坪上。
3. 当你收到飞机已着陆，请停止一切操作的反馈后，结束所有操作。

# 可用的飞行决策工具
- landing_ctrl_decision: 着陆控制决策(必须配合飞行操作工具使用)
- pid_ctrl_operations: 下滑道控制决策（必须单独使用）

# 可用的飞行操作工具flight_operations
- move_descend(time_s): 下降
- move_forward_and_descend(time_s): 向前水平移动并下降  


# 重要
- 注意：调用pid_ctrl后，无需单独调用飞行操作工具。
- 必须：调用landing_ctrl工具获得飞行决策后，再根据飞行决策调用相应飞行操作工具。
- 必须：停机坪在当前视野之外时，一定要调用move_descend(time_s)操作，而不是水平移动去寻找停机坪（危险）
- time_s: 工具的执行时间，单位为秒。
"""

    TOOLS = [
        ("tools/aircraft/msfs2024tools/flight_operations.py", "flight_operations"),
        ("tools/aircraft/flight_brain/landing_control.py", "landing_ctrl_decision"),
        ("tools/aircraft/flight_brain/pid_control.py", "pid_ctrl_operations"),
    ]
    
    MCP_SERVER_NAME = "flightbrain_agent"
    
    TOOL_DESCRIPTION = """Intelligent Flight Brain Agent for eVTOL aircraft management.
    
    This agent reads safety assessment results, analyzes images, and executes appropriate flight operations.
    
    Parameters:
        query: str - Flight management query, can include:
            - Image analysis: "Image:/path/to/image.png"
            - Flight commands based on safety assessment
            
    Returns:
        str - Flight operation execution result with print content
        
    Example queries:
        - "Image:/home/bld/dyx/FractFlow-Aircraft/tools/aircraft/sam/tmp/test_boundary.png"
    """
    
    @classmethod
    def create_config(cls):
        """Custom configuration for FlightBrain Agent"""
        from FractFlow.infra.config import ConfigManager
        from dotenv import load_dotenv
        
        load_dotenv()
        return ConfigManager(
            provider='qwen',
            qwen_model='qwen-max',
            max_iterations=50,
            custom_system_prompt=cls.SYSTEM_PROMPT,
            tool_calling_version='turbo'
        )

if __name__ == "__main__":
    FlightBrain_Agent.main()