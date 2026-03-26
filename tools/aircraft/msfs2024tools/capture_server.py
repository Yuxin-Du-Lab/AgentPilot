import requests  # 导入requests库，用于发送HTTP请求
from datetime import datetime  # 导入datetime模块，用于获取当前时间
import time  # 导入time模块，用于延时
import os  # 导入os模块，用于文件和路径操作
from dotenv import load_dotenv  # 导入dotenv库，用于加载环境变量
load_dotenv()  # 加载环境变量
API_URL_CAMERA = os.getenv('API_URL_CAMERA')  # 从环境变量中获取相机API端点URL

def capture_camera_image(image_save_path='./tmp/screenshots/current_view.png', 
                         api_url=API_URL_CAMERA):
    """
    从API端点获取相机图像并保存到指定路径
    
    参数:
        image_save_path: 图像保存路径，默认为'./tmp/screenshots/current_view.png'
        api_url: API端点URL，默认为API_URL_CAMERA
    
    返回:
        bool: 成功返回True，失败返回False
    """
    try:
        # 发送GET请求获取图像数据，stream=True允许流式接收大文件
        response = requests.get(api_url, stream=True, timeout=10)
        
                # 检查响应状态码是否为200（成功）
        if response.status_code == 200:
            # 获取当前时间并格式化为字符串
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建临时文件路径（在同一目录下，添加.tmp后缀）
            temp_filename = image_save_path + '.tmp'
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            
            # 以二进制写入模式打开临时文件
            with open(temp_filename, 'wb') as f:
                # 按块读取响应内容并写入临时文件，减少内存使用
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # 文件完全写入后，原子性地重命名为最终文件名
            # 这个操作是原子性的，可以避免读取到不完整的文件
            os.rename(temp_filename, image_save_path)
            
            # 打印成功信息
            print(f"[{timestamp}] Image saved successfully as {image_save_path}")
            return True
        else:
            # 打印错误信息
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        # 捕获网络请求异常
        print(f"Request error: {e}")
        return False
    except IOError as e:
        # 捕获文件写入异常
        print(f"File I/O error: {e}")
        return False


if __name__ == '__main__':
    print("开始定时捕获相机图像，每1秒执行一次...")
    print("按 Ctrl+C 停止程序")
    
    try:
        while True:
            # 调用函数捕获图像
            capture_camera_image()
            
            # 等待1秒
            time.sleep(1)
            
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断信号
        print("\n程序已停止")


# from flask import Flask, request, jsonify
# import os
# from datetime import datetime
# import uuid
# from PIL import Image
# import io
# import base64
# from dotenv import load_dotenv

# load_dotenv()
# app = Flask(__name__)



# # Configure upload directory
# UPLOAD_FOLDER = './tmp/screenshots'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)


# @app.route('/upload', methods=['POST'])
# def upload_screenshot():
#     try:
#         # Check if image data exists in request
#         if 'image' not in request.json:
#             return jsonify({'error': 'Image data not found'}), 400
        
#         # Get base64 encoded image data
#         image_data = request.json['image']
        
#         # Decode base64 image data
#         try:
#             # Remove data:image prefix if present
#             if ',' in image_data:
#                 image_data = image_data.split(',')[1]
            
#             # Decode base64 data
#             image_bytes = base64.b64decode(image_data)
            
#             # Open image using PIL
#             image = Image.open(io.BytesIO(image_bytes))
            
#             # Generate unique filename
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             unique_id = str(uuid.uuid4())[:8]
#             # filename = f'screenshot_{timestamp}_{unique_id}.png'
#             filename = 'current_view.png'
#             filepath = os.path.join(UPLOAD_FOLDER, filename)
            
#             # Atomic write: save to temp file first, then move atomically
#             temp_filename = f'temp_{unique_id}.png'
#             temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
            
#             # Save as PNG format to temp file
#             image.save(temp_filepath, 'PNG')
            
#             # Move to target file atomically (this is an atomic operation)
#             os.rename(temp_filepath, filepath)
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Image uploaded successfully, video updated',
#                 'filename': filename,
#                 'path': filepath,
#                 'size': f'{image.width}x{image.height}'
#             }), 200
            
#         except Exception as e:
#             return jsonify({'error': f'Image processing failed: {str(e)}'}), 400
            
#     except Exception as e:
#         return jsonify({'error': f'Server error: {str(e)}'}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'server running', 'upload_folder': UPLOAD_FOLDER}), 200

# if __name__ == '__main__':
#     print(f"Server starting...")
#     print(f"Image save directory: {os.path.abspath(UPLOAD_FOLDER)}")
#     print(f"Server address: http://{os.getenv('CAPTURE_SERVER_IP')}:{os.getenv('CAPTURE_SERVER_PORT')}")
#     print(f"Upload endpoint: http://{os.getenv('CAPTURE_SERVER_IP')}:{os.getenv('CAPTURE_SERVER_PORT')}/upload")
#     app.run(host=os.getenv('CAPTURE_SERVER_IP'), port=int(os.getenv('CAPTURE_SERVER_PORT')), debug=True)
#     # app.run(host='0.0.0.0', port=7001, debug=True)
