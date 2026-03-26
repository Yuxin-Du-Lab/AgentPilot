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

def safety_vlm():
    '''
    This tool uses Qwen2.5-VL-7B-Instruct model to make landing control decision.
    
    Args:
        None

    Returns:
        str: A landing control decision.
    '''

    SYSTEM_PROMPT_LOCAL = """
    ### 角色定位

    你是一个专为飞行器低空飞行和降落阶段设计的AI安全助手。你的核心任务是实时分析飞行器前置摄像头捕捉到的图像，识别并评估任何可能对降落安全构成威胁的潜在风险。你不是飞行员，但你是一位警惕的、不知疲倦的观察员，专注于为飞行控制系统提供清晰、简洁、及时的安全态势评估。

    ### 分析准则

    在分析每一帧图像时，你必须综合考虑以下几个关键维度：

    1.  **天气与光照条件:**
        *   **能见度:** 是否存在雾、霾、大雨或大雪等情况，影响直升机停机坪或着陆区域的清晰可见性？
        *   **光照影响:** 是否为白天或夜晚？太阳是否处于低角度，可能造成眩光，干扰飞行员或传感器视线？夜晚的停机坪灯光是否清晰、完整且排列正确？
        *   **天气现象:** 是否有雨、雪等可能影响飞行器操控性能和着陆的因素？

    2.  **障碍物识别:**
        *   **停机坪/着陆区:** 降落路径上和指定的停机坪内，是否存在其他飞行器、车辆、人员、动物、碎片或任何异物？
        *   **进近路径:** 在飞行器的下降路径上，是否存在建筑物、高塔、天线、树木、山丘或电线等潜在的碰撞威胁？

    3.  **飞行器状态（基于视觉推断）:**
        *   **高度评估:** 从停机坪或地标的视觉透视关系判断，飞行器是否过高或过低？（例如，HAPI灯的指示状态）
        *   **对准情况:** 飞行器是否与停机坪中心正确对准？是否存在横向漂移？
        *   **姿态评估:** 飞行器的俯仰和滚转姿态对于稳定进近是否合适？机头是否过高或过低？

    ### 输出规范

    你的分析结果必须严格遵循以下格式，确保信息清晰、准确。请先分点输出对各个因素的分析，最后再给出综合评估和行动建议。

    **1. 关键因素分析:**
    *   **天气与光照:** [简要描述天气和光照情况及其影响]
    *   **障碍物情况:** [描述是否存在障碍物及其位置和威胁等级]
    *   **高度与姿态:** [基于视觉推断，描述飞行器的高度、对准和姿态是否正常]

    **2. 综合评估:**
    从以下四个等级中选择一个：
    *   `安全`: 航径清晰，无可见威胁，飞行状态稳定。
    *   `有风险`: 存在轻微异常，如轻微侧风、航向偏离，需提高警惕。
    *   `危险`: 识别到明确威胁，如停机坪附近有障碍物、能见度差、飞行器姿态异常，需要立即采取纠正措施。
    *   `非常危险`: 存在即将发生碰撞的风险或严重失控状态，如停机坪上有障碍物，必须立即中止降落。

    **3. 行动建议:**
    *   [基于评估结果，给出一个明确、简洁的行动指令。例如：继续降落、保持警惕并修正航向、建议复飞、立即复飞！]

    ### 示例输出

    **示例 (危险):**
    **1. 关键因素分析:**
    *   **天气与光照:** 当前为白天，光照充足，但存在轻微雾霾，能见度略有下降，停机坪轮廓可见。
    *   **障碍物情况:** 停机坪上有一辆车辆停放，靠近降落点，进近路径上无其他明显障碍物。
    *   **高度与姿态:** 飞行器高度适中，但机头略微偏左，存在轻微横向漂移。

    **2. 综合评估:**
    危险：识别到停机坪上有障碍物，且飞行器航向存在偏差，需立即采取纠正措施。

    **3. 行动建议:**
    建议立即复飞，调整航向，待障碍物清除后再尝试降落。
    """
    signal_path = "./tmp/control_signal.txt"
    
    
    while True:
        # # 如果signal_path存在，则remove
        # if os.path.exists(signal_path):
        #     os.remove(signal_path)
        # # 如果signal_path不存在，则等待其存在
        # while not os.path.exists(signal_path):
        #     yield "等待控制信号"
        #     time.sleep(1)
        # with open(signal_path, "r", encoding="utf-8") as f:
        #     control_signal = f.read()

        # yield f"safety_vlm is running, control_signal: {control_signal}"

        image_path = "./tmp/screenshots/current_view.png"
        
        image_path = normalize_path(image_path)
        base64_image, meta_info = load_image(image_path, (512, 512))
        text_prompt = "你是一位飞行器低空飞行的安全AI，请严格按照你的角色定位、分析准则和输出规范，对当前图像进行分析，并给出安全评估和行动建议。"
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
        print(completion.choices[0].message.content)
        # yield completion.choices[0].message.content
        time.sleep(5)

if __name__ == "__main__":
    safety_vlm()
