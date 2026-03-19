import requests
from dotenv import load_dotenv
import os
load_dotenv()
API_URL_GET = os.getenv('API_URL_GET')

def get_gps_heading():
    # try:
    # 发送GET请求到API
    response = requests.get(API_URL_GET)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析JSON响应
        data = response.json()

    # 打印所有参数
    # print("⻜机状态参数:")
    state_dict = {}
    for param in data:
        # print(f"{param['name']}: {param['val']} {param['unit']} (可写: {param['writable']})")
        state_dict[param['name']] = param

    # print(state_dict['PLANE_LATITUDE'])
    # print(state_dict['PLANE_LONGITUDE'])
    # print(state_dict['PLANE_ALTITUDE'])

    # return str gps
    gps_str = f"LONGITUDE:{state_dict['PLANE_LONGITUDE']['val']}, LATITUDE:{state_dict['PLANE_LATITUDE']['val']}, ALTITUDE:{state_dict['PLANE_ALTITUDE']['val']}"
    heading_str = f'True Heading:{state_dict["PLANE_HEADING_DEGREES_TRUE"]["val"]}, Magnetic Heading:{state_dict["PLANE_HEADING_DEGREES_MAGNETIC"]["val"]}'
    return gps_str, heading_str

def get_is_on_ground():
    response = requests.get(API_URL_GET)

    if response.status_code == 200:
        data = response.json()

    state_dict = {}
    for param in data:
        state_dict[param['name']] = param

    is_on_ground = int(state_dict['SIM_ON_GROUND']['val']) > 0.5
    return is_on_ground
