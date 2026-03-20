# Agent-Pilot: A Visual Hybrid Intelligence Closed-Loop Framework for eVTOL Autonomous Landing

This repository is prepared for running **Agent-Pilot** via `tools/aircraft`.

```bash
git clone https://github.com/Yuxin-Du-Lab/AgentPilot.git
```

## Environment

### Config .env

Copy `.env.example` to `.env` in the repo root and fill in your own values:

```bash
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


