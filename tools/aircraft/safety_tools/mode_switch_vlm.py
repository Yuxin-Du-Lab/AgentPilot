from openai import OpenAI
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from aircraft.utils.image_utils import normalize_path, load_image
import os
import json
import time
# Load environment variables
load_dotenv()

def stop_PID_control_vlm(image_path="./tmp/tracked_view.png", text_prompt="请判断当前是否应该继续PID控制"):
    SYSTEM_PROMPT_VLM = """
你是一个飞行器自动降落系统的控制切换决策助手。当前系统采用两阶段降落策略:
1. PID阶段: 负责将飞机从远处引导至停机坪附近区域(粗略定位)
2. VLM阶段: 负责最后精准降落至停机坪上(精确定位)

【当前视角说明】
- 这是飞机第一视角画面,摄像机位于机舱最前端
- 画面底部可能有横梁,其余部分显示飞机正前方视野
- 你能看到停机坪及其相对位置关系

【你的任务】
根据当前画面综合判断是否应该继续使用PID控制，还是应该切换到VLM精准降落控制。

【切换条件考虑因素】
1. 距离因素: 停机坪在画面中是否足够大、足够清晰,飞机是否已接近停机坪
2. 视觉清晰度: 停机坪的标识、边界、中心点是否清晰可见
3. 位置关系: 飞机是否大致对准停机坪,偏离角度是否在可接受范围
4. 高度判断: 从停机坪在视野中的位置判断高度是否适合进入精准降落阶段
5. 安全性: 当前姿态是否稳定,视野中是否有障碍物

【判断标准】
- 如果停机坪距离较远、占据画面比例较小、或位置关系不明确,说明仍需PID引导,回复: "是"
- 如果停机坪已占据画面较大比例、轮廓清晰、位置关系明确、飞机基本对准目标,说明应该切换到VLM精准控制,回复: "否"

【重要】回复逻辑说明:
- 回复"是" = 继续PID控制 (停机坪较远,需要PID引导)
- 回复"否" = 停止PID控制,切换到VLM控制 (停机坪足够近,可以精准降落)

【回复格式】
请仅回复以下两种之一:
- "是" (表示继续PID控制)
- "否" (表示停止PID控制,切换到VLM控制)
    """
    
    image_path = normalize_path(image_path)
    base64_image, meta_info = load_image(image_path, (512, 512))
    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "system","content": SYSTEM_PROMPT_VLM},
            {"role": "user","content": [
                {"type": "text","text": text_prompt},
                {"type": "image_url",
                "image_url": {"url": f'data:image/png;base64,{base64_image}'}}
                ]}]
    )   
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content