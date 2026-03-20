# # 保存 MSFS2024 摄像机参数到 JSON 文件
# # 用法: python save_camera_params.py [--x 0] [--y 0] [--z 10] [--pitch 0] [--roll 0] [--yaw 0] [-o camera_params.json]

# import json
# import sys
# import argparse


# def save_camera_params(params, output_path="./camera_params.json"):
#     """保存摄像机参数到 JSON 文件"""
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(params, f, ensure_ascii=False, indent=2)
#     print(f"摄像机参数已保存到: {output_path}", file=sys.stderr)
#     print(f"参数: {params}", file=sys.stderr)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="保存 MSFS2024 摄像机参数到 JSON 文件")
#     parser.add_argument("--x", type=float, default=0.0, help="摄像机 X 位置")
#     parser.add_argument("--y", type=float, default=0.0, help="摄像机 Y 位置（高度）")
#     parser.add_argument("--z", type=float, default=0.0, help="摄像机 Z 位置")
#     parser.add_argument("--pitch", type=float, default=0.0, help="俯仰角")
#     parser.add_argument("--roll", type=float, default=0.0, help="横滚角")
#     parser.add_argument("--yaw", type=float, default=0.0, help="偏航角")
#     parser.add_argument("-o", "--output", default="./camera_params.json", help="输出文件路径")
#     args = parser.parse_args()

#     params = {
#         "x": args.x,
#         "y": args.y,
#         "z": args.z,
#         "pitch": args.pitch,
#         "roll": args.roll,
#         "yaw": args.yaw,
#     }
#     save_camera_params(params, args.output)

# 从 JSON 文件恢复 MSFS2024 摄像机参数
# 用法: python load_camera_params.py [camera_params.json]

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


def get_camera_params():
    """从 API 获取摄像机参数"""
    try:
        response = requests.get(API_URL_CAMERA_CTRL)
        if response.status_code == 200:
            print("摄像机参数获取成功", file=sys.stderr)
            return response.json()
        else:
            print(f"请求失败，状态码: {response.status_code}", file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    params = get_camera_params()
    print(params)
