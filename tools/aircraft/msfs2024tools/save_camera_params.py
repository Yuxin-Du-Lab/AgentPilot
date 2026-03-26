# # Save MSFS2024 camera parameters to a JSON file
# # Usage: python save_camera_params.py [--x 0] [--y 0] [--z 10] [--pitch 0] [--roll 0] [--yaw 0] [-o camera_params.json]

# import json
# import sys
# import argparse


# def save_camera_params(params, output_path="./camera_params.json"):
#     """Save camera parameters to a JSON file."""
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(params, f, ensure_ascii=False, indent=2)
#     print(f"Camera parameters saved to: {output_path}", file=sys.stderr)
#     print(f"Parameters: {params}", file=sys.stderr)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Save MSFS2024 camera parameters to a JSON file")
#     parser.add_argument("--x", type=float, default=0.0, help="Camera X position")
#     parser.add_argument("--y", type=float, default=0.0, help="Camera Y position (height)")
#     parser.add_argument("--z", type=float, default=0.0, help="Camera Z position")
#     parser.add_argument("--pitch", type=float, default=0.0, help="Pitch angle")
#     parser.add_argument("--roll", type=float, default=0.0, help="Roll angle")
#     parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle")
#     parser.add_argument("-o", "--output", default="./camera_params.json", help="Output file path")
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

# Restore MSFS2024 camera parameters from a JSON file
# Usage: python load_camera_params.py [camera_params.json]

# Restore MSFS2024 camera parameters from a JSON file
# Usage: python load_camera_params.py [camera_params.json]

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
    """Fetch camera parameters from the API."""
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
