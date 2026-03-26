import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from aircraft.utils.logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

from dotenv import load_dotenv


from PIL import Image
import base64
import io
import json
import numpy as np

load_dotenv()

import torch

from aircraft.safety_tools.Video_Depth_Anything.video_depth_anything.video_depth import VideoDepthAnything
from aircraft.safety_tools.Video_Depth_Anything.utils.dc_utils import read_video_frames

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt_path = "./tmp/video_depth_anything_vitl.pth"
# 配置模型参数
model_config = {
    'encoder': 'vitl',
    'features': 256,
    'out_channels': [256, 512, 1024, 1024]
}

# 初始化和加载模型
video_depth_anything = VideoDepthAnything(**model_config)
video_depth_anything.load_state_dict(
    torch.load(ckpt_path, map_location=DEVICE),
    strict=True
)
video_depth_anything = video_depth_anything.to(DEVICE).eval()

def estimate_video_depth(video_path):
    """
    从视频中估计深度信息
    
    参数:
        video_path (str): 输入视频的路径
        input_size (int): 处理时的输入尺寸
        max_res (int): 最大分辨率
    
    返回:
        depths: 深度图序列
    """
    
    
    # 读取视频帧
    frames, target_fps = read_video_frames(video_path, process_length=-1, target_fps=-1, max_res=1280)
    
    # 进行深度估计
    depths, _ = video_depth_anything.infer_video_depth(
        frames,
        target_fps,
        device=DEVICE,
        fp32=False
    )
    
    return depths


def normalize_path(path: str) -> str:
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info

def parse_result(text: str):
    content = text.find("方框内容识别")
    level = text.find("评估结果")
    reasoning = text.find("核心理由")
    result = {
        "complete": 0,
        "landing_spot_type": "",
        "safety_level": "",
        "safety_reasoning": ""
    }
    if content == -1 or level == -1 or reasoning == -1:
        return result
    else:
        result["complete"] = 1
    for i in range(content+6, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["landing_spot_type"] += text[i]
    result["landing_spot_type"] = result["landing_spot_type"][:-2]
    for i in range(level+4, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["safety_level"] += text[i]
    result["safety_level"] = result["safety_level"][:-2]
    for i in range(reasoning+4, len(text)):
        if text[i] == "-" or text[i] == "#":
            break
        if text[i] != "：" and text[i] != ":":
            result["safety_reasoning"] += text[i]
    result["safety_reasoning"] = result["safety_reasoning"][:-2]
    return result

def stop_PID_control() -> str:
    # return "保持PID控制"
    vid_path = "./tmp/key_frames_video.mp4"
    
    try:
        # 检查视频文件是否存在
        if not os.path.exists(vid_path):
            logger.warning(f"视频文件不存在: {vid_path}")
            return "是,PID控制正在运行"
            
        # 检查视频文件大小
        if os.path.getsize(vid_path) == 0:
            logger.warning(f"视频文件大小为0: {vid_path}")
            return "是,PID控制正在运行"
            
        logger.info("开始估算视频深度")
        depths = estimate_video_depth(vid_path)
        depths = np.array(depths)
        # min max norm it
        d_min, d_max = depths.min(), depths.max()
        depths = ((depths - d_min) / (d_max - d_min) * 255)
        logger.info(f"深度数组形状: {depths.shape}, 长度: {len(depths)}, 单帧形状: {depths[0].shape}")

        bbox_json_path = "./tmp/key_frames_video_bbox.json"
        # 计算bbox的中心点的depth
        with open(bbox_json_path, "r") as f:
            bbox_list = json.load(f)
        
        # center_depths = []
        avg_depths = []
        for frame_idx, frame_data in enumerate(bbox_list):
            # 获取当前帧的深度图
            depth = depths[frame_idx]
            
            # 获取bbox信息并计算中心点
            bbox = frame_data["bbox"]
            img_height, img_width = depth.shape
            
            # 计算bbox内所有像素的平均深度值
            bbox_x1 = int(bbox["x"] * img_width)
            bbox_y1 = int(bbox["y"] * img_height)
            bbox_x2 = int((bbox["x"] + bbox["width"]) * img_width)
            bbox_y2 = int((bbox["y"] + bbox["height"]) * img_height)
            
            # 确保bbox坐标在有效范围内
            bbox_x1 = max(0, min(bbox_x1, img_width - 1))
            bbox_y1 = max(0, min(bbox_y1, img_height - 1))
            bbox_x2 = max(0, min(bbox_x2, img_width - 1))
            bbox_y2 = max(0, min(bbox_y2, img_height - 1))
            
            # 提取bbox区域的深度值并计算平均值
            bbox_depth = depth[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
            avg_depth = np.mean(bbox_depth)
            avg_depths.append(float(avg_depth))

        
        logger.warning(f"平均深度值: {avg_depths}")
        # diff_ratio = abs(center_depths[-1] - center_depths[0]) / (center_depths[0] + 1e-6)

        # 检查avg_depths是否为空
        if not avg_depths:
            return "是,深度数据为空"

        # 计算差异比率时增加更大的epsilon值
        diff_ratio = abs(min(avg_depths) - max(avg_depths)) / (max(avg_depths) + 1e-3)
        # 0.70 for rain
        # 0.50 for reset
        if diff_ratio > 0.50:
            result = f"否,深度值变化率：{diff_ratio*100:.2f}%"
        else:
            result = f"是,深度值变化率：{diff_ratio*100:.2f}%"
        print(result)
        return result

    except Exception as e:
        logger.error(f"处理视频时发生错误: {str(e)}")
        return "是,视频处理出错"
