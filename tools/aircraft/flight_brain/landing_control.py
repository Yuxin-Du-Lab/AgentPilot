import json
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
import builtins
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, 'file': kwargs.get('file', sys.stderr)})
from aircraft.utils.image_utils import normalize_path, load_image
from aircraft.utils.self_logging import get_my_logger

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("landing_ctrl_decision")

@mcp.tool()
async def landing_ctrl():
    '''
    This tool uses Qwen-VL-Max model to make landing control decision.
    
    Args:
        None

    Returns:
        str: A landing control decision.
    '''

    SYSTEM_PROMPT_LOCAL = """

​    **1. 角色与核心任务**

​    你是一个负责将飞机降落到直升机停机坪上的AI。你的核心任务是自主操控这台多旋翼飞行器，并将其精准、安全地降落在指定平台上。

​    你输入信息是来自飞行器前置摄像头拍摄的前方的实时图像。需要注意图像中的**直升机停机坪；飞机的机体部分（白色横梁）：位于图像的最下方，占据了画面底部的一部分，是飞机驾驶舱的前部，属于机身的一部分。**

​    你的职责是分析图像，先输出对当前飞机的机体部分（白色横梁）与直升机停机坪的相对位置的综合描述，然后输出下一步要执行的单一飞行操作指令。
    


​    **2. 可用操作指令 (Action Space)**

​    你只能从以下指令中选择一个进行输出。对于所有带有 `(time_s)` 参数的指令，根据当前情况**自主决定一个最合理的持续时间（秒）** 是你的关键职责之一。

        * `move_forward_and_descend(time_s)`: 向前水平移动并下降。
        * `move_descend(time_s)`: 垂直下降。
    
    ** 3. 决策逻辑与思考框架 **
    - 【核心】如果你看不到直升机停机坪，说明你处于停机坪的正上方，你**一定**要执行:*’move_descend(time_s)'操作，让飞机降落到停机坪上。而不是尝试使用其他工具寻找停机坪
    - 当停机坪大部分在飞机的机体部分（白色横梁）的前方时，你需要执行:*’move_forward_and_descend(time_s)'操作，让机体部分（白色横梁）靠近停机坪。

​   
    **4. 输出格式（中文）**
    解释：[一段解释本文，解释当前飞机的机体部分（白色横梁）与停机坪的相对位置，以及你当前的决策阶段]
    操作：[一个单一的飞行操作指令]

    # 硬性规则
    - 直升机停机坪有H标志、红十字标志、白十字标志或其他特定几何图案
    - 非停机坪（屋顶、公路、草地、停车场等）
​    """

    with open("./tmp/control_signal.txt", "w", encoding="utf-8") as f:
        f.write("Landing has control")

    image_path = "./tmp/screenshots/current_view.png"
    
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))
    text_prompt = "请分析图片进行飞行决策"
    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system","content": SYSTEM_PROMPT_LOCAL},
            {"role": "user","content": [
                {"type": "text","text": text_prompt},
                {"type": "image_url",
                "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
                ]}]
    )
    text_result = completion.choices[0].message.content
    my_logger = get_my_logger()
    # my_logger.info(text_result)
    my_logger.record_vlm_decision(text_result)
    return json.dumps({
        'image_info': text_result,
    }, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import logging
    logging.basicConfig(stream=sys.stderr, force=True)
    print("Landing control is running")
    mcp.run(transport='stdio')
