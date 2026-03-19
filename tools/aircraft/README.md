# Agent-Pilot: tools/aircraft

## What this is

`tools/aircraft` is the Agent-Pilot implementation built on top of **FractFlow**, providing an end-to-end pipeline for **infrastructure-free autonomous landing** in **Microsoft Flight Simulator 2024** (monocular vision only).

## Architecture (high level)

- **msfs2024tools**: simulator control + image capture server
- **sam_tools**: Segment Anything server for initial target selection / segmentation
- **tracking_tools**: tracking (e.g., CSRT) for continuous target tracking
- **safety_tools**: video depth estimation + other perception utilities
- **flight_brain**: PID control and landing logic
- **safety_system**: independent safety checks / VLM-based safety reasoning

## Related work & citations

- Segment Anything (SAM): see `tools/aircraft/sam_tools/segment-anything/` for license and citation.
- Video Depth Anything: see `tools/aircraft/safety_tools/Video_Depth_Anything/` for license and citation.

## Setup

### Base env

```bash
cd path/to/FractFlow-main/tools/aircraft

uv venv
source .venv/bin/activate
uv pip install flask openai pyyaml loguru dotenv mcp pillow replicate websocket json-repair tokencost gradio scipy
uv pip install opencv-python pycocotools matplotlib onnxruntime onnx
uv pip install torch torchvision torchaudio
uv pip install einops easydict
uv pip install watchdog opencv-contrib-python==4.11.0.86

cd path/to/FractFlow-main
uv pip install -e .
```

### Deploy SAM server

Download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth to ./FractFlow-main/tools/aircraft/tmp/sam_vit_b_01ec64.pth

```bash
# terminal 1
export CUDA_VISIBLE_DEVICES="AS NEED"

cd path/to/FractFlow-main/tools/aircraft/sam_tools
git clone https://github.com/facebookresearch/segment-anything.git
cd path/to/FractFlow-main/tools/aircraft
source .venv/bin/activate
uv pip install -e ./sam_tools/segment-anything
python -m sam_tools.sam_server
```

### Deploy Video Depth Estimation

```bash
cd path/to/FractFlow-main/tools/aircraft/safety_tools
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
mv Video-Depth-Anything Video_Depth_Anything
```



```python
# modify Video_Depth_Anything/video_depth_anything/video_depth.py Line 27
# from 
utils.util import compute_scale_and_shift, get_interpolate_frames
# to
..utils.util import compute_scale_and_shift, get_interpolate_frames
```

Download https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth to ./FractFlow-main/tools/aircraft/tmp/video_depth_anything_vitl.pth

## Deploy capture image server

```bash
# terminal 2
cd path/to/FractFlow-main/tools/aircraft
python -m msfs2024tools.capture_server
# python msfs2024tools/capture_server.py
```

### Config .env

Copy `.env.example` to `.env` in the repo root and fill your own API keys.

```shell
# .env
QWEN_API_KEY=your_key

GRADIO_SERVER_IP=127.0.0.1
GRADIO_SERVER_PORT=7000

CAPTURE_SERVER_IP=0.0.0.0
CAPTURE_SERVER_PORT=7001

SAM_SERVER_IP=127.0.0.1
SAM_SERVER_PORT=7002

```

## Run

### Run gradio

```bash
# terminal 3
cd path/to/FractFlow-main/tools/aircraft
python app.py
```

### Run flight agent

```bash
cd path/to/FractFlow-main/tools/aircraft/
python flight_agent.py --query "fly"
```



