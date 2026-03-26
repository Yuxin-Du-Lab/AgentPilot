import requests
from dotenv import load_dotenv
import os
load_dotenv()
API_URL_GET = os.getenv('API_URL_GET')

def get_gps_heading():
    # try:
    # Send a GET request to the API
    response = requests.get(API_URL_GET)

    # Check the response status code
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

    # Print all parameters
    # print("Aircraft state parameters:")
    state_dict = {}
    for param in data:
        # print(f"{param['name']}: {param['val']} {param['unit']} (writable: {param['writable']})")
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
