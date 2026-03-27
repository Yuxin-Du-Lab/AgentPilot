import cv2
import numpy as np
import os
import time
import json
from PIL import Image, ImageDraw
from sam_tools.sam_tool import SAM_TOOL
import asyncio  # Added
from utils.image_utils import atomic_dump_json, atomic_write_cv2_image


def save_frames_to_video(frames, bbox_datas, video_len, output_path):
    """
    Save a list of frames as a video while keeping only the lower half of each image.
    
    Args:
        frames: List of image frames
        bbox_datas: Corresponding list of bbox data
        video_len: Desired video length in frames
        output_path: Output video path
    """
    if len(frames) == 0:
        return
        
    # Uniformly sample frames if the frame count exceeds video_len
    if len(frames) > video_len:
        indices = np.linspace(0, len(frames)-1, video_len, dtype=int)
        frames = [frames[i] for i in indices]
        bbox_datas = [bbox_datas[i] for i in indices]
    
    # Deep-copy bbox data and adjust the y coordinate
    adjusted_bbox_datas = []
    for bbox_data in bbox_datas:
        adjusted_data = bbox_data.copy()
        # Adjust y: subtract 0.5 (top half) and multiply by 2 because the height is halved
        adjusted_data['bbox'] = adjusted_data['bbox'].copy()
        y = adjusted_data['bbox']['y']
        height = adjusted_data['bbox']['height']
        
        if y >= 0.5:  # If the bbox is in the lower half
            adjusted_data['bbox']['y'] = (y - 0.5) * 2
            adjusted_data['bbox']['height'] = height * 2
        else:  # If it is in the upper half, mark it as invalid
            adjusted_data['bbox']['y'] = -1
            adjusted_data['bbox']['height'] = 0
            
        adjusted_bbox_datas.append(adjusted_data)
    
    # Save the adjusted bbox data
    bbox_json_path = output_path.replace('.mp4', '_bbox.json')
    atomic_dump_json(adjusted_bbox_datas, bbox_json_path, indent=2)
    
    # Get the first frame size
    height, width = frames[0].shape[:2]
    
    # Crop each frame to keep only the lower half
    cropped_frames = [frame[height//2:, :] for frame in frames]
    
    # Get the cropped frame size
    new_height = height // 2
    
    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1, (width, new_height))
    
    # Write frames
    for frame in cropped_frames:
        out.write(frame)
    
    out.release()

def init_tracker(initial_image_path, initial_mask_path):
    """
    Initialize the tracker and return the tracker object plus the initial bounding box.
    Args:
        initial_image_path: Initial image path
        initial_mask_path: Initial mask path
    Returns:
        tracker: Initialized CSRT tracker (or None)
        bbox: Bounding box (x, y, w, h) (or None)
        img_shape: Image dimensions (height, width) (or None)
        msg: Status message string
    """
    if not os.path.exists(initial_image_path):
        return None, None, None, f"初始图片未找到: {initial_image_path}"
    if not os.path.exists(initial_mask_path):
        return None, None, None, f"初始mask未找到: {initial_mask_path}"

    # Read the initial image and mask
    initial_image = cv2.imread(initial_image_path)
    initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)

    if initial_image is None or initial_mask is None:
        return None, None, None, "无法读取初始图片或mask文件"

    # Process the mask
    initial_mask[initial_mask > 0] = 1

    # Extract a bounding box from the mask
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, "mask中未找到有效轮廓"

    # Use the largest contour as the tracking target
    largest_contour = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest_contour)

    # Verify that the bounding box stays within image bounds
    x, y, w, h = bbox
    img_height, img_width = initial_image.shape[:2]
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        # Clip the bounding box to the image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        bbox = (x, y, w, h)

    # Validate the bounding-box dimensions
    if w <= 0 or h <= 0:
        return None, None, None, f"无效的边界框尺寸: width={w}, height={h}"

    # Create the CSRT tracker
    tracker = None
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            return None, None, None, "无法创建CSRT跟踪器，请检查OpenCV版本"

    if tracker is None:
        return None, None, None, "跟踪器创建失败"

    # Ensure the bounding box uses floating-point values
    bbox_float = tuple(float(v) for v in bbox)

    # Initialize the tracker
    success = tracker.init(initial_image, bbox_float)
    if success is None:
        success = True
    if not success:
        return None, None, None, "跟踪器初始化失败"

    return tracker, bbox, (img_height, img_width), "跟踪器初始化成功"

def continue_tracking(should_continue_func=None, video_len=20):
    """
    Continuously track the target by starting from the initial image and mask,
    then repeatedly reading new frames from new_image_path.
    
    Args:
        should_continue_func: Function that returns True to continue and False to stop
        video_len: Desired video length in frames
        
    Yields:
        str: Tracking status messages
    """
    initial_image_path = "./tmp/init_tracking_view.png"
    initial_mask_path = "./tmp/init_tracking_mask.png"
    # Fixed input and output paths
    new_image_path = "./tmp/screenshots/current_view.png"
    output_path = "./tmp/tracked_view.png"
    video_output_path = "./tmp/key_frames_video.mp4"
    # Remove the existing video output file if it exists
    if os.path.exists(video_output_path):
        os.remove(video_output_path)
    sam_tool = SAM_TOOL()
    new_image = None
    tracked_bbox = None
    try:
        # Initialize via init_tracker
        tracker, tracked_bbox, img_shape, msg = init_tracker(initial_image_path, initial_mask_path)
        if tracker is None:
            yield msg
            return
        yield msg
        x, y, w, h = tracked_bbox
        img_height, img_width = img_shape

        # Store frames and bbox data for video generation
        frames = []
        bbox_datas = []
        
        # Loop counter
        loop_count = 0

        # Run only once if should_continue_func is not provided
        if should_continue_func is None:
            should_continue = False
        else:
            should_continue = True
            
        while should_continue:
            loop_count += 1  # Increment once per loop

            # Reinitialize the tracker every 10 loops
            if loop_count % 30 == 0:
                # Compute the bounding-box center
                if new_image is not None and tracked_bbox is not None:
                    x, y, w, h = tracked_bbox
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    points = (center_x, center_y)
                    # Call sam_tool.detect with the previous frame and center point
                    asyncio.run(sam_tool.detect(new_image, points))
                tracker, tracked_bbox, img_shape, msg = init_tracker(initial_image_path, initial_mask_path)
                if tracker is None:
                    yield f"第{loop_count}次循环重新初始化失败: {msg}"
                    break
                else:
                    yield f"第{loop_count}次循环已重新初始化跟踪器"
            
            # Check whether tracking should continue
            if should_continue_func and not should_continue_func():
                yield f"追踪已停止（第{len(frames)}帧）"
                break
            
            # Check whether the next image exists
            if not os.path.exists(new_image_path):
                yield f"第{len(frames)+1}帧：新图片未找到，等待..."
                time.sleep(0.5)
                continue
            
            # Read the next image
            new_image = cv2.imread(new_image_path)
            if new_image is None:
                yield f"第{len(frames)+1}帧：无法读取新图片，跳过此帧"
                time.sleep(0.5)
                continue
            
            # Update the tracker
            success, tracked_bbox = tracker.update(new_image)
            
            if not success:
                yield f"第{len(frames)+1}帧：跟踪失败，尝试继续..."
                time.sleep(0.5)
                fail_img = np.zeros((224, 224, 3), dtype=np.uint8)
                crop_image_path = "./tmp/tracked_view_crop.png"
                atomic_write_cv2_image(crop_image_path, fail_img)
                json_output_path = "./tmp/tracked_view_bbox.json"
                fail_data = {
                    "frame": -1,
                    "bbox": {
                        "x": -1,
                        "y": -1,
                        "width": -1,
                        "height": -1
                    },
                    "timestamp": time.time()
                }
                atomic_dump_json(fail_data, json_output_path, indent=2)
                break
            
            # Draw the bounding box and save the image
            result_image = new_image.copy()
            x, y, w, h = [int(v) for v in tracked_bbox]
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            atomic_write_cv2_image(output_path, result_image)
            # Crop result_image around the bbox center; crop size is 1.2x the bbox and never exceeds image bounds
            # Compute the crop region centered on the bounding box
            crop_scale = 2
            crop_w = int(w * crop_scale)
            crop_h = int(h * crop_scale)
            center_x = x + w // 2
            center_y = y + h // 2

            img_h, img_w = result_image.shape[:2]

            # Compute the top-left and bottom-right crop coordinates
            crop_x1 = max(0, center_x - crop_w // 2)
            crop_y1 = max(0, center_y - crop_h // 2)
            crop_x2 = min(img_w, center_x + crop_w // 2)
            crop_y2 = min(img_h, center_y + crop_h // 2)

            # Adjust the crop region if it exceeds image bounds
            if crop_x2 - crop_x1 != crop_w:
                if crop_x1 == 0:
                    crop_x2 = min(img_w, crop_x1 + crop_w)
                else:
                    crop_x1 = max(0, crop_x2 - crop_w)
            if crop_y2 - crop_y1 != crop_h:
                if crop_y1 == 0:
                    crop_y2 = min(img_h, crop_y1 + crop_h)
                else:
                    crop_y1 = max(0, crop_y2 - crop_h)

            crop_image = result_image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_image_path = output_path.replace('.png', '_crop.png')

            atomic_write_cv2_image(crop_image_path, crop_image)
            
            # Save bounding-box coordinates as a JSON file
            bbox_data = {
                "frame": len(frames) + 1,
                "bbox": {
                    "x": x / img_width,
                    "y": y / img_height,
                    "width": w / img_width,
                    "height": h / img_height
                },
                "timestamp": time.time()
            }
            
            # Record the image and bbox data once per second (moved here)
            frames.append(new_image.copy())
            bbox_datas.append(bbox_data)  # Store bbox data for the current frame
            
            # Save the video and bbox data once the frame count reaches video_len
            if len(frames) >= video_len:
                yield f"正在保存视频，当前帧数：{len(frames)}"
                save_frames_to_video(frames, bbox_datas, video_len, video_output_path)
                yield f"视频已保存到：{video_output_path}"
            
            # Build the JSON path corresponding to the image path
            json_output_path = output_path.replace('.png', '_bbox.json')
            atomic_dump_json(bbox_data, json_output_path, indent=2)

            yield f"第{len(frames)+1}帧跟踪成功 - 位置: ({x}, {y}, {w}, {h})"
            
            time.sleep(0.5)
            
        # One-shot execution is not supported
        if not should_continue_func:
            raise ValueError("不支持一次性执行")
            
        # Save the last video and bbox data before exiting
        if frames:
            yield "正在保存最终视频..."
            save_frames_to_video(frames, bbox_datas, video_len, video_output_path)
            yield f"最终视频已保存到：{video_output_path}"
            
    except Exception as e:
        yield f"跟踪过程中出现异常: {str(e)}"
