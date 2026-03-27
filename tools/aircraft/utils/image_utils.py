import os
import base64
import io
import json
import tempfile
from PIL import Image
from typing import Tuple

def normalize_path(path: str) -> str:
    # Expand ~ to user's home directory
    expanded_path = os.path.expanduser(path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded_path):
        expanded_path = os.path.abspath(expanded_path)
        
    return expanded_path

def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info

def _create_atomic_temp_path(path: str) -> str:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    stem, suffix = os.path.splitext(os.path.basename(path))
    fd, temp_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=suffix, dir=directory)
    os.close(fd)
    return temp_path

def _cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def atomic_write_bytes(data: bytes, path: str) -> None:
    normalized_path = normalize_path(path)
    temp_path = _create_atomic_temp_path(normalized_path)
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
        os.replace(temp_path, normalized_path)
    except Exception:
        _cleanup_temp_file(temp_path)
        raise

def atomic_save_pil_image(image: Image.Image, path: str, **save_kwargs) -> None:
    normalized_path = normalize_path(path)
    temp_path = _create_atomic_temp_path(normalized_path)
    try:
        image.save(temp_path, **save_kwargs)
        os.replace(temp_path, normalized_path)
    except Exception:
        _cleanup_temp_file(temp_path)
        raise

def atomic_write_cv2_image(path: str, image, params: list[int] | None = None) -> None:
    import cv2

    normalized_path = normalize_path(path)
    suffix = os.path.splitext(normalized_path)[1]
    if not suffix:
        raise ValueError(f"Image path must include an extension: {path}")

    encode_args = params if params is not None else []
    ok, buffer = cv2.imencode(suffix, image, encode_args)
    if not ok:
        raise OSError(f"Failed to encode image for {path}")

    atomic_write_bytes(buffer.tobytes(), normalized_path)

def atomic_dump_json(data, path: str, **json_kwargs) -> None:
    payload = json.dumps(data, **json_kwargs).encode("utf-8")
    atomic_write_bytes(payload, path)

def load_image_eager(path: str, mode: str | None = None) -> Image.Image:
    normalized_path = normalize_path(path)
    with Image.open(normalized_path) as image:
        image.load()
        if mode is not None and image.mode != mode:
            image = image.convert(mode)
        return image.copy()
