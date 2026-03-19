# Contributors: Yuxin Du, Chenglin Liu, Yunqi Zhao

import cv2
import numpy as np
import os
import time
import json
from PIL import Image, ImageDraw
from sam_tools.sam_tool import SAM_TOOL
import asyncio  # 新增


def save_frames_to_video(frames, bbox_datas, video_len, output_path):
    """
    将帧列表保存为视频，只保留图像下半部分
    
    Args:
        frames: 图片帧列表
        bbox_datas: 对应的bbox数据列表
        video_len: 期望的视频长度（帧数）
        output_path: 输出视频路径
    """
    if len(frames) == 0:
        return
        
    # 如果帧数超过video_len，进行等距采样
    if len(frames) > video_len:
        indices = np.linspace(0, len(frames)-1, video_len, dtype=int)
        frames = [frames[i] for i in indices]
        bbox_datas = [bbox_datas[i] for i in indices]
    
    # 创建bbox_datas的深拷贝，并调整y坐标
    adjusted_bbox_datas = []
    for bbox_data in bbox_datas:
        adjusted_data = bbox_data.copy()
        # 调整y坐标：减去0.5（上半部分），然后乘以2（因为高度减半）
        adjusted_data['bbox'] = adjusted_data['bbox'].copy()
        y = adjusted_data['bbox']['y']
        height = adjusted_data['bbox']['height']
        
        if y >= 0.5:  # 如果在下半部分
            adjusted_data['bbox']['y'] = (y - 0.5) * 2
            adjusted_data['bbox']['height'] = height * 2
        else:  # 如果在上半部分，将其设为无效值
            adjusted_data['bbox']['y'] = -1
            adjusted_data['bbox']['height'] = 0
            
        adjusted_bbox_datas.append(adjusted_data)
    
    # 保存调整后的bbox数据
    bbox_json_path = output_path.replace('.mp4', '_bbox.json')
    with open(bbox_json_path, 'w') as f:
        json.dump(adjusted_bbox_datas, f, indent=2)
    
    # 获取第一帧的尺寸
    height, width = frames[0].shape[:2]
    
    # 裁剪每一帧，只保留下半部分
    cropped_frames = [frame[height//2:, :] for frame in frames]
    
    # 获取裁剪后的尺寸
    new_height = height // 2
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1, (width, new_height))
    
    # 写入帧
    for frame in cropped_frames:
        out.write(frame)
    
    out.release()

def init_tracker(initial_image_path, initial_mask_path):
    """
    初始化跟踪器，返回tracker对象和初始边界框
    Args:
        initial_image_path: 初始图片路径
        initial_mask_path: 初始mask路径
    Returns:
        tracker: 初始化好的CSRT跟踪器（或None）
        bbox: 边界框 (x, y, w, h)（或None）
        img_shape: 图片尺寸 (height, width)（或None）
        msg: 状态信息字符串
    """
    if not os.path.exists(initial_image_path):
        return None, None, None, f"初始图片未找到: {initial_image_path}"
    if not os.path.exists(initial_mask_path):
        return None, None, None, f"初始mask未找到: {initial_mask_path}"

    # 读取初始图片和mask
    initial_image = cv2.imread(initial_image_path)
    initial_mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)

    if initial_image is None or initial_mask is None:
        return None, None, None, "无法读取初始图片或mask文件"

    # 处理mask
    initial_mask[initial_mask > 0] = 1

    # 从mask中提取边界框
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, "mask中未找到有效轮廓"

    # 找到最大的轮廓作为跟踪目标
    largest_contour = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest_contour)

    # 验证边界框是否在图像范围内
    x, y, w, h = bbox
    img_height, img_width = initial_image.shape[:2]
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        # 裁剪边界框到图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        bbox = (x, y, w, h)

    # 验证边界框尺寸
    if w <= 0 or h <= 0:
        return None, None, None, f"无效的边界框尺寸: width={w}, height={h}"

    # 创建CSRT跟踪器
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

    # 确保边界框是浮点数格式
    bbox_float = tuple(float(v) for v in bbox)

    # 初始化跟踪器
    success = tracker.init(initial_image, bbox_float)
    if success is None:
        success = True
    if not success:
        return None, None, None, "跟踪器初始化失败"

    return tracker, bbox, (img_height, img_width), "跟踪器初始化成功"

def continue_tracking(should_continue_func=None, video_len=20):
    """
    持续跟踪函数，使用初始图片和mask开始跟踪，然后不断从new_image_path读取新图片进行跟踪
    
    Args:
        should_continue_func: 一个函数，返回True时继续追踪，返回False时停止
        video_len: 期望的视频长度（帧数）
        
    Yields:
        str: 追踪状态信息
    """
    initial_image_path = "./tmp/init_tracking_view.png"
    initial_mask_path = "./tmp/init_tracking_mask.png"
    # 固定的新图片路径和输出路径
    new_image_path = "./tmp/screenshots/current_view.png"
    output_path = "./tmp/tracked_view.png"
    video_output_path = "./tmp/key_frames_video.mp4"
    # 检查video_output_path是否存在文件，若存在，则删除
    if os.path.exists(video_output_path):
        os.remove(video_output_path)
    sam_tool = SAM_TOOL()
    new_image = None
    tracked_bbox = None
    try:
        # 使用init_tracker函数初始化
        tracker, tracked_bbox, img_shape, msg = init_tracker(initial_image_path, initial_mask_path)
        if tracker is None:
            yield msg
            return
        yield msg
        x, y, w, h = tracked_bbox
        img_height, img_width = img_shape

        # 添加帧列表和bbox数据列表用于记录视频
        frames = []
        bbox_datas = []
        
        # 新增循环计数器
        loop_count = 0

        # 如果没有提供should_continue_func，只执行一次
        if should_continue_func is None:
            should_continue = False
        else:
            should_continue = True
            
        while should_continue:
            loop_count += 1  # 每次循环+1

            # 每10次循环重新初始化tracker
            if loop_count % 30 == 0:
                # 计算bounding box中心点
                if new_image is not None and tracked_bbox is not None:
                    x, y, w, h = tracked_bbox
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    points = (center_x, center_y)
                    # 调用sam_tool.detect，传入上一帧图片和中心点
                    asyncio.run(sam_tool.detect(new_image, points))
                tracker, tracked_bbox, img_shape, msg = init_tracker(initial_image_path, initial_mask_path)
                if tracker is None:
                    yield f"第{loop_count}次循环重新初始化失败: {msg}"
                    break
                else:
                    yield f"第{loop_count}次循环已重新初始化跟踪器"
            
            # 检查是否应该继续
            if should_continue_func and not should_continue_func():
                yield f"追踪已停止（第{len(frames)}帧）"
                break
            
            # 检查新图片是否存在
            if not os.path.exists(new_image_path):
                yield f"第{len(frames)+1}帧：新图片未找到，等待..."
                time.sleep(0.5)
                continue
            
            # 读取新图片
            new_image = cv2.imread(new_image_path)
            if new_image is None:
                yield f"第{len(frames)+1}帧：无法读取新图片，跳过此帧"
                time.sleep(0.5)
                continue
            
            # 更新跟踪器
            success, tracked_bbox = tracker.update(new_image)
            
            if not success:
                yield f"第{len(frames)+1}帧：跟踪失败，尝试继续..."
                time.sleep(0.5)
                fail_img = np.zeros((224, 224, 3), dtype=np.uint8)
                crop_image_path = "./tmp/tracked_view_crop.png"
                cv2.imwrite(crop_image_path, fail_img)
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
                with open(json_output_path, 'w') as f:
                    json.dump(fail_data, f, indent=2)
                break
            
            # 绘制边界框并保存图片
            result_image = new_image.copy()

            
            # 保存带边界框的图片
            cv2.imwrite(output_path, result_image)
            x, y, w, h = [int(v) for v in tracked_bbox]
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
            # crop result_image，最大不超过原图，x, y分开计算，crop后的图片中心为bounding box的中心，且图片大小为bounding box的1.2倍
            # 计算crop区域，中心为bounding box中心，大小为1.2倍的bounding box，且不超过原图边界
            crop_scale = 2
            crop_w = int(w * crop_scale)
            crop_h = int(h * crop_scale)
            center_x = x + w // 2
            center_y = y + h // 2

            img_h, img_w = result_image.shape[:2]

            # 计算crop区域左上角和右下角坐标
            crop_x1 = max(0, center_x - crop_w // 2)
            crop_y1 = max(0, center_y - crop_h // 2)
            crop_x2 = min(img_w, center_x + crop_w // 2)
            crop_y2 = min(img_h, center_y + crop_h // 2)

            # 如果crop区域超出图片边界，调整crop区域
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

            cv2.imwrite(crop_image_path, crop_image)
            
            # 保存边界框坐标为json文件
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
            
            # 每1秒记录一次图片和bbox数据（移到这里）
            frames.append(new_image.copy())
            bbox_datas.append(bbox_data)  # 现在保存的是当前帧的bbox数据
            
            # 当帧数达到或超过video_len时，保存视频和bbox数据
            if len(frames) >= video_len:
                yield f"正在保存视频，当前帧数：{len(frames)}"
                save_frames_to_video(frames, bbox_datas, video_len, video_output_path)
                yield f"视频已保存到：{video_output_path}"
            
            # 生成json文件路径（与图片路径对应）
            json_output_path = output_path.replace('.png', '_bbox.json')
            with open(json_output_path, 'w') as f:
                json.dump(bbox_data, f, indent=2)

            yield f"第{len(frames)+1}帧跟踪成功 - 位置: ({x}, {y}, {w}, {h})"
            
            time.sleep(0.5)
            
        # 如果是一次性执行
        if not should_continue_func:
            raise ValueError("不支持一次性执行")
            
        # 函数结束前保存最后一次视频和bbox数据
        if frames:
            yield "正在保存最终视频..."
            save_frames_to_video(frames, bbox_datas, video_len, video_output_path)
            yield f"最终视频已保存到：{video_output_path}"
            
    except Exception as e:
        yield f"跟踪过程中出现异常: {str(e)}"
