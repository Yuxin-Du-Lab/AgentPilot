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
        """Initialize the SAM model on the server."""
        try:
            response = requests.post(f"{self.server_url}/initialize")
            result = response.json()
            print(f"初始化结果: {result['message']}")
            return result['status'] == 'success'
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def check_health(self):
        """Check the server health status."""
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
        Send an image segmentation request.
        
        Args:
            image_path: Path to the image file
            prompt_points: List of prompt points [[x1, y1], [x2, y2], ...]
            prompt_labels: List of prompt labels [1, 0, 1, ...] (1 for foreground, 0 for background)
        """
        try:
            # Encode the image
            image_base64 = self.encode_image(image)
            
            # Prepare the request payload
            data = {
                "image": image_base64,
                "prompt_points": prompt_points,
                "prompt_labels": prompt_labels or [1] * len(prompt_points)
            }
            
            # Send the request
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
        """Decode a base64-encoded mask into a NumPy array."""
        mask_bytes = base64.b64decode(mask_base64)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        return mask



def sam_request(client: SAMClient, image, prompt_points):
    # Check the server health status
    if not client.check_health():
        print("服务器不可用，请先启动服务器")
        return
    
    # Initialize the SAM model
    if not client.initialize_server():
        print("模型初始化失败")
        return

    # Run segmentation
    mask = client.segment_image(image, prompt_points, None)
    return mask
