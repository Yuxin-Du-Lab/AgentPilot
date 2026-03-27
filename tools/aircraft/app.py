import gradio as gr
import numpy as np
from PIL import Image
import os
import time
import threading

from sam_tools.sam_tool import SAM_TOOL
from utils.image_utils import load_image_eager

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
        
        # Add tracking control state
        self.tracking_enabled = False
        self.tracking_status = "None"  # Store tracking status text
        self.tracking_generator = None  # Store the tracking generator
        self.tracker_btn_text = "Start Tracking"  # Track the button label state

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
            # Load the initial image
            self.current_image = self.load_frame(self.img_path)
            
            # Add the timer component
            timer = gr.Timer(value=2.0)  # Trigger once every 2 seconds
            
            with gr.Row():
                with gr.Column():
                    self.image_display = gr.Image(
                        label="Click on the target vertipad",
                        value=self.current_image,
                        interactive=True,
                    )
            with gr.Row():
                with gr.Column():   
                    # Image display components
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

                    
            
            # Bind events
            self.image_display.select(
                fn=self.sam_tool.detect,
                inputs=[self.image_display],
                outputs=[self.coordinate_info]
            )

            self.reload_btn.click(
                fn=self.auto_update_all,
                inputs=None,
                outputs=[self.image_display, self.tracker_image, self.image_to_vlm, self.image_sam, self.tracking_info]
            )
            
            # Bind the timer event to refresh images and tracking info
            timer.tick(
                fn=self.auto_update_all,
                inputs=[self.auto_update_checkbox],
                outputs=[self.image_display, self.tracker_image, self.image_to_vlm, self.image_sam, self.tracking_info]
            )
            
            # Bind the tracking button to toggle tracking state
            self.tracker_btn.click(
                fn=self.toggle_tracking_with_button,
                inputs=None,
                outputs=[self.tracker_btn, self.tracking_info]
            )

        self.interface = interface

    def toggle_tracking_with_button(self):
        """Toggle continuous tracking on or off via the button."""
        self.tracking_enabled = not self.tracking_enabled
        
        if self.tracking_enabled:
            # Start the background tracking thread
            threading.Thread(target=self.background_tracking, daemon=True).start()
            self.tracking_status = "Tracking started..."
            self.tracker_btn_text = "Stop Tracking"
        else:
            self.tracking_status = "Tracking stopped"
            self.tracker_btn_text = "Start Tracking"
            # Delete tracking images
            try:
                if os.path.exists(self.tracker_img_path):
                    os.remove(self.tracker_img_path)
                if os.path.exists(self.tracker_img_path_vlm):
                    os.remove(self.tracker_img_path_vlm)
            except Exception as e:
                print(f"Failed to remove tracking images: {e}")
        
        return gr.Button(value=self.tracker_btn_text), self.tracking_status

    def toggle_tracking(self, enabled):
        """Enable or disable continuous tracking (kept for compatibility)."""
        self.tracking_enabled = enabled
        if enabled:
            # Start the background tracking thread
            threading.Thread(target=self.background_tracking, daemon=True).start()
            self.tracking_status = "Tracking started..."
            return self.tracking_status
        else:
            self.tracking_status = "Tracking stopped"
            return self.tracking_status
    
    def background_tracking(self):
        """Background function for continuous tracking."""
        try:
            self.tracking_generator = continue_tracking(self.tracking_enabled_check)
            for status_msg in self.tracking_generator:
                if not self.tracking_enabled:
                    break
                # Update the status variable
                self.tracking_status = status_msg
                time.sleep(0.1)  # Avoid updating too frequently
        except Exception as e:
            self.tracking_status = f"Tracking error: {str(e)}"
    
    def tracking_enabled_check(self):
        """Check whether tracking should continue."""
        return self.tracking_enabled
    
    def load_sam_image(self):
        """Load the SAM image."""
        initial_image_path = "./tmp/init_tracking_view.png"
        initial_mask_path = "./tmp/init_tracking_mask.png"
        if not os.path.exists(initial_image_path) or not os.path.exists(initial_mask_path):
            return None
        try:
            initial_image = load_image_eager(initial_image_path)
            initial_mask = load_image_eager(initial_mask_path)
            return self.sam_tool.overlay_mask_on_image(initial_image, initial_mask)
        except (OSError, ValueError) as e:
            print(f"Failed to load SAM image: {e}")
            return None

    def load_tracker_image(self):
        """Load the tracking image."""
        try:
            if os.path.exists(self.tracker_img_path) and os.path.exists(self.tracker_img_path_vlm):
                tracker_image = load_image_eager(self.tracker_img_path)
                tracker_crop = load_image_eager(self.tracker_img_path_vlm)
                return tracker_image, tracker_crop
            else:
                return None, None
        except Exception as e:
            print(f"Failed to load tracking image: {e}")
            return None, None

    def auto_update_all(self, auto_update_enabled):
        """Automatically update all UI components."""
        # Update the main image
        if auto_update_enabled:
            main_image = self.load_frame(self.img_path)
        else:
            main_image = gr.update()
        
        # Update tracking images
        tracker_image, image_to_vlm = self.load_tracker_image()
        sam_image = self.load_sam_image()
        
        # Update tracking info
        current_tracking_info = self.tracking_status
        
        return main_image, tracker_image, image_to_vlm, sam_image, current_tracking_info

    def load_frame(self, img_pth):
        # Check whether the file exists
        if not os.path.exists(img_pth):
            return None
        # Load the image
        try:
            img = load_image_eager(img_pth)
        except (OSError, ValueError) as e:
            print(f"Failed to load frame: {e}")
            return None
        # Handle different image formats
        if img.mode == 'RGBA':
            # Convert RGBA to RGB
            background = Image.new('RGB', img.size, (255, 255, 255))  # white background
            background.paste(img, mask=img.split()[-1])  # use the alpha channel as a mask
            img = background
        elif img.mode not in ['RGB', 'L']:
            # Convert other formats to RGB
            img = img.convert('RGB')
        # Store as a NumPy array
        img = np.array(img)
        self.img = img
        return img

    def launch(self):
        self.interface.queue().launch(share=self.share, server_name=self.server_name, server_port=self.server_port)

# Launch the app
if __name__ == "__main__":
    my_gradio = Gradio_Interface()
    my_gradio.launch()
