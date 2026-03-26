from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()
import requests

from PIL import Image
import base64
import io
import cv2
import numpy as np

class SAMClient:
    def __init__(self, server_url="http://localhost:7002"):
        self.server_url = server_url
        self.initialize_server()
        
    def initialize_server(self):
        """初始化服务器上的SAM模型"""
        try:
            response = requests.post(f"{self.server_url}/initialize")
            result = response.json()
            print(f"初始化结果: {result['message']}")
            return result['status'] == 'success'
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def check_health(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health")
            result = response.json()
            print(f"服务器状态: {result}")
            return result['status'] == 'healthy'
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False

    def encode_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_image
    
    def segment_image(self, image, prompt_points, prompt_labels=None):
        """
        发送图像分割请求
        
        Args:
            image_path: 图像文件路径
            prompt_points: 提示点列表 [[x1, y1], [x2, y2], ...]
            prompt_labels: 提示点标签列表 [1, 0, 1, ...] (1为前景，0为背景)
        """
        try:
            # 编码图像
            image_base64 = self.encode_image(image)
            
            # 准备请求数据
            data = {
                "image": image_base64,
                "prompt_points": prompt_points,
                "prompt_labels": prompt_labels or [1] * len(prompt_points)
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/segment",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            result = response.json()
            
            if result['status'] == 'success':
                print(f"分割成功，得分: {result['score']}")
                return self.decode_mask(result['mask'])
            else:
                print(f"分割失败: {result['message']}")
                return None
                
        except Exception as e:
            print(f"请求失败: {e}")
            return None
    
    def decode_mask(self, mask_base64):
        """将base64编码的mask解码为numpy数组"""
        mask_bytes = base64.b64decode(mask_base64)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        return mask



def sam_request(client: SAMClient, image, prompt_points):
    # 检查服务器健康状态
    if not client.check_health():
        print("服务器不可用，请先启动服务器")
        return
    
    # 初始化SAM模型
    if not client.initialize_server():
        print("模型初始化失败")
        return

    # 执行分割
    mask = client.segment_image(image, prompt_points, None)
    return mask
