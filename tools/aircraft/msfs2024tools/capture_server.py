import requests  # Import requests for sending HTTP requests
from datetime import datetime  # Import datetime for generating timestamps
import time  # Import time for delays
import os  # Import os for file and path operations
from dotenv import load_dotenv  # Import dotenv for loading environment variables
load_dotenv()  # Load environment variables
API_URL_CAMERA = os.getenv('API_URL_CAMERA')  # Read the camera API endpoint from the environment

def capture_camera_image(image_save_path='./tmp/screenshots/current_view.png', 
                         api_url=API_URL_CAMERA):
    """
    Fetch a camera image from the API endpoint and save it to the target path.
    
    Args:
        image_save_path: Path for saving the image, defaults to './tmp/screenshots/current_view.png'
        api_url: Camera API endpoint, defaults to API_URL_CAMERA
    
    Returns:
        bool: True on success, False on failure
    """
    try:
        # Send a GET request for the image data; stream=True helps with large responses
        response = requests.get(api_url, stream=True, timeout=10)
        
                # Check whether the response status code indicates success
        if response.status_code == 200:
            # Generate a formatted timestamp string
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a temporary file path in the same directory with a .tmp suffix
            temp_filename = image_save_path + '.tmp'
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            
            # Open the temporary file in binary write mode
            with open(temp_filename, 'wb') as f:
                # Write the response in chunks to reduce memory usage
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            # Atomically rename the fully written temp file to the final filename
            # This avoids reading a partially written file
            os.rename(temp_filename, image_save_path)
            
            # Print success information
            print(f"[{timestamp}] Image saved successfully as {image_save_path}")
            return True
        else:
            # Print error information
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        # Catch network request exceptions
        print(f"Request error: {e}")
        return False
    except IOError as e:
        # Catch file write exceptions
        print(f"File I/O error: {e}")
        return False


if __name__ == '__main__':
    print("开始定时捕获相机图像，每1秒执行一次...")
    print("按 Ctrl+C 停止程序")
    
    try:
        while True:
            # Capture an image
            capture_camera_image()
            
            # Wait for 1 second
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Handle Ctrl+C interruption
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
