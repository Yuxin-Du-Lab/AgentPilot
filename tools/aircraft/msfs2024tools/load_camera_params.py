# 从 JSON 文件恢复 MSFS2024 摄像机参数
# 用法: python load_camera_params.py [camera_params.json]

import requests
import json
import sys
import os
from dotenv import load_dotenv

load_dotenv()
API_URL_CAMERA_CTRL = os.getenv('API_URL_CAMERA_CTRL')
if API_URL_CAMERA_CTRL is None:
    raise ValueError("环境变量 API_URL_CAMERA_CTRL 未设置，请检查 .env 文件")


def load_camera_params(input_path="./camera_params.json"):
    """从 JSON 文件读取摄像机参数并通过 API 设置"""
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}", file=sys.stderr)
        return

    with open(input_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    print(f"正在设置摄像机参数: {params}", file=sys.stderr)

    try:
        response = requests.post(API_URL_CAMERA_CTRL, json=params)
        if response.status_code == 200:
            print("摄像机参数设置成功", file=sys.stderr)
        else:
            print(f"请求失败，状态码: {response.status_code}", file=sys.stderr)
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}", file=sys.stderr)


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "./camera_params.json"
    load_camera_params(input_file)
