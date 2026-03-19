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
cd path/to/FractFlow-main/tools/aircraft
python flight_agent.py --query "fly"
```


```python
# Level 1: Use default configuration
class SimpleTool(ToolTemplate):
    SYSTEM_PROMPT = "..."
    TOOL_DESCRIPTION = "..."
    # Automatically use DeepSeek default configuration

# Level 2: Partial customization
class CustomTool(ToolTemplate):
    SYSTEM_PROMPT = "..."
    TOOL_DESCRIPTION = "..."
    
    @classmethod
    def create_config(cls):
        return ConfigManager(
            provider='deepseek',           # Switch model provider
            openai_model='deepseek-reasoner',       # Specify model
            max_iterations=20           # Adjust iteration count
        )

# Level 3: Full customization
class AdvancedTool(ToolTemplate):
    SYSTEM_PROMPT = "..."
    TOOL_DESCRIPTION = "..."
    
    @classmethod
    def create_config(cls):
        from dotenv import load_dotenv
        load_dotenv()
        
        return ConfigManager(
            provider='qwen',
            anthropic_model='qwen-plus',
            max_iterations=50,
            temperature=0.7,
            custom_system_prompt=cls.SYSTEM_PROMPT + "\nAdditional instructions...",
            tool_calling_version='turbo',
            timeout=120
        )
```

## Tool Showcase

### Core Tools

#### File I/O Agent - File Operation Expert
```bash
# Basic file operations
python tools/core/file_io/file_io_agent.py --query "Read config.json file"
python tools/core/file_io/file_io_agent.py --query "Write 'Hello World' to output.txt"

# Advanced operations
python tools/core/file_io/file_io_agent.py --query "Read lines 100-200 of large file data.csv"
python tools/core/file_io/file_io_agent.py --query "Delete all lines containing 'ERROR' from temp.log"
```

**Feature Highlights**:
- Intelligent file operations: read, write, delete, insert
- Large file chunked processing
- Line-level precise operations
- Automatic directory creation

#### GPT Imagen Agent - AI Image Generation
```bash
# Image generation
python tools/core/gpt_imagen/gpt_imagen_agent.py --query "Generate image: save_path='spring_garden.png' prompt='a beautiful spring garden with flowers'"
python tools/core/gpt_imagen/gpt_imagen_agent.py --query "Generate image: save_path='robot.png' prompt='futuristic robot illustration'"
```

#### Web Search Agent - Web Search
```bash
# Web search
python tools/core/websearch/websearch_agent.py --query "Latest AI technology developments"
python tools/core/websearch/websearch_agent.py --query "Python performance optimization methods"
```

#### Weather Agent - Weather Query Assistant (US Only)
```bash
# Weather queries (US cities only)
python tools/core/weather/weather_agent.py --query "Weather in New York today"
python tools/core/weather/weather_agent.py --query "5-day forecast for San Francisco"
```

This tool can only query weather information within the United States.

#### Visual Question Answer - Visual Q&A
```bash
# Image understanding (based on Qwen-VL-Plus model)
python tools/core/visual_question_answer/vqa_agent.py --query "Image: /path/to/image.jpg What objects are in this picture?"
python tools/core/visual_question_answer/vqa_agent.py --query "Image: /path/to/photo.png Describe the scene in detail"
```

### Composite Tools

#### Visual Article Agent - Illustrated Article Generator

This is a typical representative of fractal intelligence, coordinating multiple tools to generate complete text-image content:

```bash
# Generate complete illustrated articles
python tools/composite/visual_article_agent.py --query "Write an article about AI development with illustrations"

# Customized articles
python tools/composite/visual_article_agent.py --query "设定：一个视觉识别AI统治社会的世界，人类只能依赖它解释图像。主人公却拥有“人类视觉直觉”，并因此被怀疑为异常个体。
要求：以第一人称，写一段剧情片段，展现他与AI模型对图像理解的冲突。
情绪基调：冷峻、怀疑、诗性。"
```


![](assets/visual_article.gif)

**Workflow**:
1. Analyze article requirements and structure
2. Use `file_manager_agent` to write chapter content
3. Use `image_creator_agent` to generate supporting illustrations
4. Integrate into complete Markdown document

**Output Example**:
```
output/visual_article_generator/ai_development/
├── article.md           # Complete article
└── images/             # Supporting images
    ├── section1-fig1.png
    ├── section2-fig1.png
    └── section3-fig1.png
```

#### Web Save Agent - Intelligent Web Saving
```bash
# Intelligent web saving (fractal intelligence example)
python tools/composite/web_save_agent.py --query "Search for latest Python tutorials and save to a comprehensive guide file"
python tools/composite/web_save_agent.py --query "Find information about machine learning and create an organized report"
```

**Feature Highlights**:
- Fractal intelligence combining web search and file saving
- Intelligent content organization and structuring
- Automatic file path management
- Multi-round search support

## API Reference

### Two Tool Development Approaches

FractFlow provides two flexible tool development approaches to meet different development needs:

#### Approach 1: Inherit ToolTemplate (Recommended)

Standardized tool development approach with automatic support for three running modes:

```python
from FractFlow.tool_template import ToolTemplate

class MyTool(ToolTemplate):
    """Standard tool class"""
    
    SYSTEM_PROMPT = """Your tool behavior instructions"""
    TOOL_DESCRIPTION = """Tool functionality description"""
    
    # Optional: depend on other tools
    TOOLS = [("path/to/tool.py", "tool_name")]
    
    @classmethod
    def create_config(cls):
        return ConfigManager(...)

# Automatically supports three running modes
# python my_tool.py                    # MCP server mode
# python my_tool.py --interactive      # Interactive mode  
# python my_tool.py --query "..."      # Single query mode
```

#### Approach 2: Custom Agent Class

Completely autonomous development approach:

```python
from FractFlow.agent import Agent
from FractFlow.infra.config import ConfigManager

async def main():
    # Custom configuration
    config = ConfigManager(
        provider='deepseek',
        deepseek_model='deepseek-chat',
        max_iterations=5
    )
    
    # Create Agent
    agent = Agent(config=config, name='my_agent')
    
    # Manually add tools
    agent.add_tool("./tools/weather/weather_mcp.py", "forecast_tool")
    
    # Initialize and use
    await agent.initialize()
    result = await agent.process_query("Your query")
    await agent.shutdown()
```

### ToolTemplate Base Class

FractFlow's core base class providing unified tool development framework:

```python
class ToolTemplate:
    """FractFlow tool template base class"""
    
    # Required attributes
    SYSTEM_PROMPT: str      # Agent system prompt
    TOOL_DESCRIPTION: str   # Tool functionality description
    
    # Optional attributes
    TOOLS: List[Tuple[str, str]] = []        # Dependent tools list
    MCP_SERVER_NAME: Optional[str] = None    # MCP server name
    
    # Core methods
    @classmethod
    def create_config(cls) -> ConfigManager:
        """Create configuration - can be overridden"""
        pass
    
    @classmethod
    async def create_agent(cls) -> Agent:
        """Create agent instance"""
        pass
    
    @classmethod
    def main(cls):
        """Main entry point - supports three running modes"""
        pass
```

#### Key Attribute Details

**Important Role of TOOL_DESCRIPTION**:

In FractFlow's fractal intelligence architecture, `TOOL_DESCRIPTION` is not just documentation for developers, but more importantly:

- **Reference manual for upper-layer Agents**: When a composite tool (like visual_article_agent) calls lower-layer tools, the upper-layer Agent reads the lower-layer tool's TOOL_DESCRIPTION to understand how to use it correctly
- **Tool interface specification**: Defines input parameter formats, return value structures, usage scenarios, etc.
- **Basis for intelligent calling**: Upper-layer Agents determine when and how to call specific tools based on this description

**Example**: When visual_article_agent calls file_io tool:
```python
# Upper-layer Agent reads file_io tool's TOOL_DESCRIPTION
# Then constructs call requests based on parameter formats described
TOOLS = [("tools/core/file_io/file_io_mcp.py", "file_operations")]
```

Therefore, writing clear and accurate TOOL_DESCRIPTION is crucial for the correct operation of fractal intelligence. However, TOOL_DESCRIPTION should not be too long.

**SYSTEM_PROMPT Writing Guidelines**:

Unlike TOOL_DESCRIPTION which faces upper-layer Agents, `SYSTEM_PROMPT` is the internal behavior instruction for the current Agent. Reference visual_article_agent's practice:

**Structured Design**:
```python
# Reference: tools/composite/visual_article_agent.py
SYSTEM_PROMPT = """
【Strict Constraints】
❌ Absolutely Forbidden: Direct content output
✅ Must Execute: Complete tasks through tool calls

【Workflow】
1. Analyze requirements
2. Call related tools
3. Verify results
"""
```

**Key Techniques**:
- **Clear Prohibitions**: Use `❌` to define what cannot be done, avoiding common errors
- **Forced Execution**: Use `✅` to specify required behavior patterns
- **Process-oriented**: Break complex tasks into clear steps
- **Verification Mechanism**: Require confirmation of results after each step

This design ensures consistency and predictability of Agent behavior, which is key to reliable operation of composite tools.

### Configuration Management

```python
from FractFlow.infra.config import ConfigManager

# Basic configuration
config = ConfigManager()

# Custom configuration
config = ConfigManager(
    provider='openai',              # Model provider: openai/anthropic/deepseek
    openai_model='gpt-4',          # Specific model
    max_iterations=20,             # Maximum iterations
    temperature=0.7,               # Generation temperature
    custom_system_prompt="...",    # Custom system prompt
    tool_calling_version='stable', # Tool calling version: stable/turbo
    timeout=120                    # Timeout setting
)
```

## File Organization
```
tools/
├── core/                 # Core tools
│   └── your_tool/
│       ├── your_tool_agent.py    # Main agent
│       ├── your_tool_mcp.py      # MCP tool implementation
│       └── __init__.py
└── composite/            # Composite tools
    └── your_composite_tool.py
```
#### Naming Conventions
- File names: `snake_case`
- Class names: `PascalCase`
