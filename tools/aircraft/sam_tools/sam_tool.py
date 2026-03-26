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
        # Mask method
        self.mask_type = mask_type # boundary, mask, bbox
        # Output size
        self.crop_size = crop_size
        # Current image
        self.img = None

    def overlay_mask_on_image(self, image, mask):
        """
        Overlay the mask on the image and save the result.
        
        Args:
            image: PIL.Image object or NumPy array containing the source image
            mask: NumPy array containing the segmentation mask
        
        Returns:
            PIL.Image: Image with the mask overlaid
        """
        # Ensure both image and mask are NumPy arrays
        img_np = np.array(image)
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            # If the mask has three channels, use the first one
            mask_np = mask_np[..., 0]
            
        # Build a colored mask (red with transparency)
        color = np.array([255, 0, 0], dtype=np.uint8)  # red
        alpha = 128  # semi-transparent
        overlay = img_np.copy()
        
        if img_np.shape[-1] == 3:
            # RGB
            overlay[mask_np > 0] = (0.5 * overlay[mask_np > 0] + 0.5 * color).astype(np.uint8)
        elif img_np.shape[-1] == 4:
            # RGBA
            overlay[mask_np > 0, :3] = (0.5 * overlay[mask_np > 0, :3] + 0.5 * color).astype(np.uint8)
        else:
            # Grayscale
            overlay[mask_np > 0] = 255

        # Save the overlay result
        overlay_img = Image.fromarray(overlay)
        return overlay_img
        
    # Core detection logic; note that this is async
    async def detect(self, image, points:gr.SelectData):
        # image = self.img
        # img_array = image.copy()
        image = Image.fromarray(image)
        # Get the click coordinates; supports gradio click data or plain [x, y]
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
        
        # Return the final result
        # self.masked_img = img_array
        # return img_array, boundary_cropped, f"Processing complete\nCurrent click coordinates: ({x}, {y})"
