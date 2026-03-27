# Agent-Pilot: A Visual Hybrid Intelligence Closed-Loop Framework for eVTOL Autonomous Landing

## Important Notes

- After each failure or fault, reset both SAM and tracking from the Gradio frontend before starting the next run.
- Consider lowering the in-game resolution in MSFS 2024 when needed. A lower resolution usually improves overall system response speed.

## MSFS 2024 Setup

Before using `tools/aircraft`, prepare the simulator and toolkit on the Windows machine that runs Microsoft Flight Simulator 2024.

### Install Microsoft Flight Simulator 2024

- Install **Microsoft Flight Simulator 2024**. This project targets MSFS 2024 rather than MSFS 2020.
- Launch MSFS 2024 once after installation and confirm the simulator can run normally,and choose the JOBY S4 eVTOL in the Free Flight Mode on a vertiport.
- Keep the IPv4 address of the MSFS 2024 machine. You will use it later in `.env` for `API_URL_CTRL`, `API_URL_CAMERA`, `API_URL_GET`, and `API_URL_CAMERA_CTRL`.

### Install and Use the MSFS24 AICtrl DevKit

- Download [MSFS24_AICtrl_DevKit_1.0.1_x64_en-US.pdf](assets/msfs/MSFS24_AICtrl_DevKit_1.0.1_x64_en-US.pdf) from this repository.
- The tracked file is the DevKit installer stored with a `.pdf` suffix. After downloading it, rename the file suffix to `.msi`.
- Run the renamed `.msi` installer on the Windows machine that runs MSFS 2024.
- After installation, use the toolkit to expose the control and image endpoints required by this project:
  - `http://<MSFS_IP>:5000/set`
  - `http://<MSFS_IP>:5000/camera_image`
  - `http://<MSFS_IP>:5000/get`
  - `http://<MSFS_IP>:5000/camera_control`
- Fill the corresponding values in `.env` before launching `capture_server` or `flight_agent`.
- Toolkit key: `U2FsdGVkX1+yIEUG4XAvTKDwcFp6vtcTiExgdVy8HKDy8nuybzmGnyMXBOKAOqD++6z/KgiuNuiLHt12IU1mig==`

This repository is prepared for running **Agent-Pilot** via `tools/aircraft`.

```bash
git clone https://github.com/Yuxin-Du-Lab/AgentPilot.git
```

## Environment

### Config .env

Copy `.env.example` to `.env` in the repo root and fill in your own values:

```bash
cd path/to/AgentPilot
cp .env.example .env
```

Then edit `.env`:

- `QWEN_API_KEY`: Your Qwen / DashScope API key, available at [DashScope Console](https://dashscope.console.aliyun.com/)
- `API_URL_CTRL` / `API_URL_CAMERA` / `API_URL_GET`: Replace `<IPv4 of MSFS2024 Computer>` with the actual IP of the machine running MSFS2024 (e.g. `192.168.1.100`)
- Other `*_IP` / `*_PORT` values can be left as defaults unless you have conflicts


### Base env

> If installation is slow, use the Tsinghua mirror:
> ```bash
> uv pip install <packages> -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

#### MSFS Camera Control

Use `path/to/msfs_camera_control.py` to set camera parameters:

```bash
cd path/to
python3 path/to/msfs_camera_control.py --host 10.7.144.111 --port 5000 --method post --x 0 --y -0.01 --z 2.27 --pitch 0 --yaw 0 --roll 0
```

```bash
cd path/to/AgentPilot/tools/aircraft

uv venv
source .venv/bin/activate
uv pip install flask openai pyyaml loguru dotenv mcp pillow replicate websocket json-repair tokencost gradio scipy
uv pip install opencv-python pycocotools matplotlib onnxruntime onnx
uv pip install torch torchvision torchaudio
uv pip install einops easydict
uv pip install watchdog opencv-contrib-python==4.11.0.86

cd path/to/AgentPilot
uv pip install -e .
```

### Deploy SAM server

Download `sam_vit_b_01ec64.pth` from the [SAM model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) and put it at:
`./tools/aircraft/tmp/sam_vit_b_01ec64.pth`

```bash
# terminal 1
export CUDA_VISIBLE_DEVICES="AS NEED"

cd path/to/AgentPilot/tools/aircraft
source .venv/bin/activate

uv pip install -e ./sam_tools/segment-anything
python -m sam_tools.sam_server
```

### Deploy Video Depth Estimation

Download `video_depth_anything_vitl.pth` from [Video Depth Anything on HuggingFace](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) and put it at:
`./tools/aircraft/tmp/video_depth_anything_vitl.pth`

> Note: `tools/aircraft/safety_tools/Video_Depth_Anything/` is vendored in this repo.

### Deploy capture image server

```bash
# terminal 2
cd path/to/AgentPilot/tools/aircraft
source .venv/bin/activate
python -m msfs2024tools.capture_server
```

## Run

### Run gradio UI

```bash
# terminal 3
cd path/to/AgentPilot/tools/aircraft
source .venv/bin/activate
python app.py
```

Once the server starts, open the Gradio URL printed in the terminal to access the web interface.

**Usage steps:**

1. Click on the target vertipad in the visual scene at the top.
2. Wait for the SAM segmentation result to appear in the panel below.
3. Confirm the target vertipad is correctly segmented, then click **Start Tracking**.
4. Verify in the tracking view that the vertipad is being tracked correctly.

### Run flight agent

```bash
# terminal 4
cd path/to/AgentPilot/tools/aircraft
source .venv/bin/activate
python flight_agent.py --query "landing on the target vertipot"
```

## Acknowledgment
Thanks to [Fractflow](https://github.com/EnVision-Research/FractFlow), [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything), and [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything).
