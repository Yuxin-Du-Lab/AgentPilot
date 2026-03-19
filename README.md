# Agent-Pilot (tools/aircraft)

This repository is prepared for running **Agent-Pilot** via `tools/aircraft`.

For the detailed guide, see: `tools/aircraft/README.md`.

## Environment

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

Download `sam_vit_b_01ec64.pth` and put it at:
`./tools/aircraft/tmp/sam_vit_b_01ec64.pth`

```bash
# terminal 1
export CUDA_VISIBLE_DEVICES="AS NEED"

cd path/to/FractFlow-main/tools/aircraft
source .venv/bin/activate

uv pip install -e ./sam_tools/segment-anything
python -m sam_tools.sam_server
```

### Deploy Video Depth Estimation

Download `metric_video_depth_anything_vitl.pth` and put it at:
`./tools/aircraft/tmp/video_depth_anything_vitl.pth`

> Note: `tools/aircraft/safety_tools/Video_Depth_Anything/` is vendored in this repo.

### Deploy capture image server

```bash
# terminal 2
cd path/to/FractFlow-main/tools/aircraft
python -m msfs2024tools.capture_server
```

### Config .env

Copy `.env.example` to `.env` in the repo root and fill your own API keys.

```env
QWEN_API_KEY=your_key

GRADIO_SERVER_IP=127.0.0.1
GRADIO_SERVER_PORT=7000

CAPTURE_SERVER_IP=0.0.0.0
CAPTURE_SERVER_PORT=7001

SAM_SERVER_IP=127.0.0.1
SAM_SERVER_PORT=7002
```

## Run

### Run gradio UI

```bash
# terminal 3
cd path/to/FractFlow-main/tools/aircraft
python app.py
```

### Run flight agent

```bash
# terminal 4
cd path/to/FractFlow-main/tools/aircraft
python flight_agent.py --query "fly"
```


