# Contributors: Yunqi Zhao, Yuxin Du

import gradio as gr
import numpy as np
from PIL import Image
import os
import time
import threading

from sam_tools.sam_tool import SAM_TOOL
from safety_tools.safety_vlm import safety_vlm

# from safety_check.safty_mcp import Safety_VLM_Local
from tracking_tools.tracking_tool import continue_tracking

class Gradio_Interface():
    def __init__(self, share=True, server_name=os.getenv('GRADIO_SERVER_IP'), server_port=int(os.getenv('GRADIO_SERVER_PORT'))):
        self.img_path = "./tmp/screenshots/current_view.png"

        # True Vars
        self.share = share
        self.server_name = server_name
        self.server_port = server_port
        self.current_image = None
        
        # 添加追踪控制变量
        self.tracking_enabled = False
        self.tracking_status = "None"  # 存储追踪状态信息
        self.tracking_generator = None  # 存储追踪生成器
        self.tracker_btn_text = "Start Tracking"  # 追踪按钮文本状态

        self.tracker_img_path = "./tmp/tracked_view.png"
        self.tracker_img_path_vlm = "./tmp/tracked_view_crop.png"

        self.sam_img_path = "./tmp/sam_segmentation.png"

        # --- init all components ---
        self.sam_tool = SAM_TOOL(mask_type="boundary", crop_size=256)
        print("SAM Component Loaded !")
        # ---------------------------

        # init gradio IO after all other components
        self._build_interface()
    
    # Gradio Interface Core
    def _build_interface(self):
        with gr.Blocks(title="Auto Landing System") as interface:
            gr.Markdown("# Visual Scene Selector")
            gr.Markdown("**External reload**: Create `reload_trigger.txt` to trigger a reload")
            # 获取初始图片
            self.current_image = self.load_frame(self.img_path)
            
            # 添加定时器组件
            timer = gr.Timer(value=2.0)  # 每2秒触发一次
            
            with gr.Row():
                with gr.Column():
                    self.image_display = gr.Image(
                        label="Click on the target vertipad",
                        value=self.current_image,
                        interactive=True,
                    )
            with gr.Row():
                with gr.Column():   
                    # 图片显示组件 1, 2
                    self.image_sam = gr.Image(
                        label="SAM Segmentation Result",
                        value=None,
                        interactive=False,
                    )
                    self.coordinate_info = gr.Textbox(
                        label="Visual Analysis",
                        value="None",
                        lines=6,
                        interactive=False
                    )
                    self.tracking_info = gr.Textbox(
                        label="Tracking Info",
                        value="None",
                        lines=6,
                        interactive=False
                    )
                    self.safety_vlm_info = gr.Textbox(
                        label="Safety Analysis",
                        value="None",
                        lines=6,
                        interactive=False
                    )
                    self.reload_btn = gr.Button("Refresh Image")
                    self.safety_vlm_btn = gr.Button("Safety VLM Analysis")
                    self.tracker_btn = gr.Button("Start Tracking")
                    self.auto_update_checkbox = gr.Checkbox(label="Enable Auto Update", value=True)
                
                with gr.Column():
                    self.tracker_image = gr.Image(
                        label="Tracking View",
                        value=None,
                        interactive=False,
                    )

                    self.image_to_vlm = gr.Image(
                        label="VLM Input View",
                        value=None,
                        interactive=False,
                    )

                    
            
            # 事件绑定
            self.image_display.select(
                fn=self.sam_tool.detect,
                inputs=[self.image_display],
                outputs=[self.coordinate_info]
            )

            self.safety_vlm_btn.click(
                fn=safety_vlm,
                inputs=None,
                outputs=[self.safety_vlm_info]
            )
            
            self.reload_btn.click(
                fn=self.auto_update_all,
                inputs=None,
                outputs=[self.image_display, self.tracker_image, self.image_to_vlm, self.image_sam, self.tracking_info]
            )
            
            # 定时器事件绑定 - 更新图片和追踪信息
            timer.tick(
                fn=self.auto_update_all,
                inputs=[self.auto_update_checkbox],
                outputs=[self.image_display, self.tracker_image, self.image_to_vlm, self.image_sam, self.tracking_info]
            )
            
            # 追踪按钮事件 - 点击切换追踪状态
            self.tracker_btn.click(
                fn=self.toggle_tracking_with_button,
                inputs=None,
                outputs=[self.tracker_btn, self.tracking_info]
            )

        self.interface = interface

    def toggle_tracking_with_button(self):
        """通过按钮控制持续追踪的开启和关闭"""
        self.tracking_enabled = not self.tracking_enabled
        
        if self.tracking_enabled:
            # 启动后台追踪线程
            threading.Thread(target=self.background_tracking, daemon=True).start()
            self.tracking_status = "Tracking started..."
            self.tracker_btn_text = "Stop Tracking"
        else:
            self.tracking_status = "Tracking stopped"
            self.tracker_btn_text = "Start Tracking"
            # 删除追踪图片
            try:
                if os.path.exists(self.tracker_img_path):
                    os.remove(self.tracker_img_path)
                if os.path.exists(self.tracker_img_path_vlm):
                    os.remove(self.tracker_img_path_vlm)
            except Exception as e:
                print(f"Failed to remove tracking images: {e}")
        
        return gr.Button(value=self.tracker_btn_text), self.tracking_status

    def toggle_tracking(self, enabled):
        """控制持续追踪的开启和关闭（保留用于兼容性）"""
        self.tracking_enabled = enabled
        if enabled:
            # 启动后台追踪线程
            threading.Thread(target=self.background_tracking, daemon=True).start()
            self.tracking_status = "Tracking started..."
            return self.tracking_status
        else:
            self.tracking_status = "Tracking stopped"
            return self.tracking_status
    
    def background_tracking(self):
        """后台持续追踪函数"""
        try:
            self.tracking_generator = continue_tracking(self.tracking_enabled_check)
            for status_msg in self.tracking_generator:
                if not self.tracking_enabled:
                    break
                # 更新状态变量
                self.tracking_status = status_msg
                time.sleep(0.1)  # 避免更新过于频繁
        except Exception as e:
            self.tracking_status = f"Tracking error: {str(e)}"
    
    def tracking_enabled_check(self):
        """检查追踪是否应该继续"""
        return self.tracking_enabled
    
    def load_sam_image(self):
        """加载SAM图片"""
        initial_image_path = "./tmp/init_tracking_view.png"
        initial_mask_path = "./tmp/init_tracking_mask.png"
        initial_image = Image.open(initial_image_path)
        initial_mask = Image.open(initial_mask_path)
        overlay_img = self.sam_tool.overlay_mask_on_image(initial_image, initial_mask)
        return overlay_img

    def load_tracker_image(self):
        """加载追踪图片"""
        try:
            if os.path.exists(self.tracker_img_path):
                return Image.open(self.tracker_img_path), Image.open(self.tracker_img_path_vlm)
            else:
                return None, None
        except Exception as e:
            print(f"Failed to load tracking image: {e}")
            return None, None

    def auto_update_all(self, auto_update_enabled):
        """自动更新所有组件的函数"""
        # 更新主图片
        if auto_update_enabled:
            main_image = self.load_frame(self.img_path)
        else:
            main_image = gr.update()
        
        # 更新追踪图片
        tracker_image, image_to_vlm = self.load_tracker_image()
        sam_image = self.load_sam_image()
        
        # 更新追踪信息
        current_tracking_info = self.tracking_status
        
        return main_image, tracker_image, image_to_vlm, sam_image, current_tracking_info

    def load_frame(self, img_pth):
        # 检测文件存在
        if not os.path.exists(img_pth):
            return None, f"Image file not found: {img_pth}"
        # 加载
        img = Image.open(img_pth)
        # 处理不同格式的图片
        if img.mode == 'RGBA':
            # 将RGBA转换为RGB
            background = Image.new('RGB', img.size, (255, 255, 255))  # 白色背景
            background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
            img = background
        elif img.mode not in ['RGB', 'L']:
            # 其他格式转换为RGB
            img = img.convert('RGB')
        # 保存到np array
        img = np.array(img)
        self.img = img
        return img

    def launch(self):
        self.interface.queue().launch(share=self.share, server_name=self.server_name, server_port=self.server_port)

# 启动应用
if __name__ == "__main__":
    my_gradio = Gradio_Interface()
    my_gradio.launch()