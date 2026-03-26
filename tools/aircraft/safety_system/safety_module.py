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

def safety_llm(text_prompt):
    SYSTEM_PROMPT_LLM = """
      你是一个飞行器低空飞行和降落阶段的AI安全助手。
    """
    client = OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system","content": SYSTEM_PROMPT_LLM},
            {"role": "user","content": [
                {"type": "text","text": text_prompt},
                ]}]
    )   
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def safety_vlm(image_path, text_prompt):
    SYSTEM_PROMPT_VLM = """
        你是一个飞行器低空飞行和降落阶段的AI安全助手。
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

if __name__ == "__main__":
    print(os.getenv('QWEN_API_KEY'))
    safety_vlm()
    safety_llm()
