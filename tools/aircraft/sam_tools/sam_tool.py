from dotenv import load_dotenv
import os
from PIL import Image
import gradio as gr
from sam_tools.sam_client import SAMClient, sam_request
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

class SAM_TOOL:
    def __init__(self, mask_type="boundary", crop_size=512):
        # SAM Client
        self.client = SAMClient()
        # 遮罩方法
        self.mask_type = mask_type # boundary, mask, bbox
        # 返回大小
        self.crop_size = crop_size
        # 当前图片
        self.img = None

    def overlay_mask_on_image(self, image, mask):
        """
        将掩码叠加到图像上并保存结果
        
        Args:
            image: PIL.Image 对象或numpy数组，原始图像
            mask: numpy数组，分割掩码
        
        Returns:
            PIL.Image: 叠加掩码后的图像
        """
        # 确保image和mask都是numpy数组
        img_np = np.array(image)
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            # 如果mask是三通道，取第一个通道
            mask_np = mask_np[..., 0]
            
        # 生成彩色mask（红色，带透明度）
        color = np.array([255, 0, 0], dtype=np.uint8)  # 红色
        alpha = 128  # 半透明
        overlay = img_np.copy()
        
        if img_np.shape[-1] == 3:
            # RGB
            overlay[mask_np > 0] = (0.5 * overlay[mask_np > 0] + 0.5 * color).astype(np.uint8)
        elif img_np.shape[-1] == 4:
            # RGBA
            overlay[mask_np > 0, :3] = (0.5 * overlay[mask_np > 0, :3] + 0.5 * color).astype(np.uint8)
        else:
            # 灰度
            overlay[mask_np > 0] = 255

        # 保存叠加结果
        overlay_img = Image.fromarray(overlay)
        return overlay_img
        
    # 核心检测 - 注意异步！
    async def detect(self, image, points:gr.SelectData):
        # image = self.img
        # img_array = image.copy()
        image = Image.fromarray(image)
        # 获取点击的坐标 - 支持点击或普通[x,y]
        if isinstance(points, gr.SelectData):
            x, y = points.index[0], points.index[1]
        else:
            x, y = points

        mask = sam_request(self.client, image, [[x, y]])

        # for tracking
        initial_image_path = "./tmp/init_tracking_view.png"
        initial_mask_path = "./tmp/init_tracking_mask.png"
        image.save(initial_image_path)
        Image.fromarray(mask).save(initial_mask_path)
        return f"处理完成\n当前点击坐标: ({x}, {y})"
        
        # 返回最终结果
        # self.masked_img = img_array
        # return img_array, boundary_cropped, f"处理完成\n当前点击坐标: ({x}, {y})"
