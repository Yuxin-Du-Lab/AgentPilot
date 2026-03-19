"""Minimal script to run `safety_vlm` on a single frame + bbox.

Example:
  python run_single_frame_vlm.py \
    --image /path/to/xxx_tracked_view.png \
    --bbox /path/to/xxx_tracked_view_bbox.json \
    --prompt "Check whether the red box is severely offset from the helipad." \
    --draw

Notes:
- Reads bbox from bbox.json (normalized coords in [0,1]: x, y, width, height).
- Optionally draws the bbox and saves *_bounding_box_tmp.png.
- Then calls safety_vlm(image_path, prompt) and prints the response.
"""
from __future__ import annotations

# ==================== 在这里填默认路径/提示词（可不传命令行参数） ====================
# 命令行参数会覆盖这里的默认值。
DEFAULT_IMAGE_PATH = ""  # set via --image


DEFAULT_BBOX_JSON_PATH = ""  # set via --bbox
# 例如: "/data2/.../tracking_logs/2026-01-20 16:35:58_tracked_view_bbox.json"

DEFAULT_PROMPT = "Describe what you see in the frame and whether it is safe to continue the landing."

SAM_IMAGE_PATH = ""  # optional, set via --sam-image
SAM_MASK_PATH = ""  # optional, set via --sam-mask





# 默认行为：与 evaluate_1.py 一致，先画框再送入 safety_vlm
DEFAULT_DRAW = True
# ============================================================================



import argparse
import json
import os
from os.path import exists

import cv2
import numpy as np

from safety_module import safety_vlm


def _load_bbox(bbox_json_path: str) -> dict:
    if not exists(bbox_json_path):
        raise FileNotFoundError(f"bbox json not found: {bbox_json_path}")
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bbox = data.get("bbox")
    if not isinstance(bbox, dict):
        raise ValueError(f"Invalid bbox json format, missing dict field 'bbox': {bbox_json_path}")

    required = ["x", "y", "width", "height"]
    missing = [k for k in required if k not in bbox]
    if missing:
        raise ValueError(f"Invalid bbox json format, missing keys {missing} in {bbox_json_path}")

    return bbox


def _draw_bbox(image_path: str, bbox: dict, out_path: str) -> str:
    if not exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image (possibly corrupted or truncated): {image_path}")

    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]  # 0~1
    h_img, w_img = image.shape[:2]

    x1 = int(x * w_img)
    y1 = int(y * h_img)
    x2 = int((x + w) * w_img)
    y2 = int((y + h) * h_img)

    # clamp
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
    ok = cv2.imwrite(out_path, image)
    if not ok:
        raise IOError(f"Failed to write image: {out_path}")
    return out_path


def _crop_middle_third_vertical(image_path: str, out_dir: str) -> str:
    """Split image width into thirds and keep the middle strip."""
    if not exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image (possibly corrupted or truncated): {image_path}")

    h, w = image.shape[:2]
    if w < 3:
        raise ValueError(f"Image width too small to split into thirds: {w}px ({image_path})")

    third = w // 3
    x1 = third
    x2 = third * 2
    cropped = image[:, x1:x2]

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(image_path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    out_path = os.path.join(out_dir, f"{root}_middle_third{ext}")

    ok = cv2.imwrite(out_path, cropped)
    if not ok:
        raise IOError(f"Failed to write cropped image: {out_path}")
    return out_path


def _crop_bbox_region(image_path: str, bbox: dict, out_dir: str) -> str:
    """Crop the image using the bbox region."""
    if not exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image (possibly corrupted or truncated): {image_path}")

    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]  # 0~1
    h_img, w_img = image.shape[:2]

    x1 = int(x * w_img)
    y1 = int(y * h_img)
    x2 = int((x + w) * w_img)
    y2 = int((y + h) * h_img)

    # clamp
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))

    # crop bbox region
    cropped = image[y1:y2, x1:x2]

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(image_path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"
    out_path = os.path.join(out_dir, f"{root}_bbox_cropped{ext}")

    ok = cv2.imwrite(out_path, cropped)
    if not ok:
        raise IOError(f"Failed to write cropped bbox image: {out_path}")
    return out_path


def _save_mask_and_masked_image(
    image_path: str,
    mask_path: str,
    out_dir: str | None = None,
    masked_out_path: str | None = None,
) -> tuple[str, str, str]:
    if not exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    if not exists(mask_path):
        raise FileNotFoundError(f"mask not found: {mask_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image (possibly corrupted or truncated): {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask (possibly corrupted or truncated): {mask_path}")

    if out_dir is None:
        out_dir = os.path.dirname(mask_path) or os.path.dirname(image_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(image_path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".png"

    mask_binary_path = os.path.join(out_dir, f"{root}_mask_binary{ext}")
    if masked_out_path:
        masked_image_path = masked_out_path
        masked_dir = os.path.dirname(masked_out_path) or "."
        os.makedirs(masked_dir, exist_ok=True)
    else:
        masked_image_path = os.path.join(out_dir, f"{root}_masked{ext}")

    mask_binary = (mask > 0).astype(np.uint8) * 255
    ok = cv2.imwrite(mask_binary_path, mask_binary)
    if not ok:
        raise IOError(f"Failed to write mask binary image: {mask_binary_path}")

    masked_image = cv2.bitwise_and(image, image, mask=mask_binary)
    ok = cv2.imwrite(masked_image_path, masked_image)
    if not ok:
        raise IOError(f"Failed to write masked image: {masked_image_path}")

    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"Mask contains no foreground pixels: {mask_path}")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    mask_cropped = image[y1 : y2 + 1, x1 : x2 + 1]
    mask_cropped_path = os.path.join(out_dir, f"{root}_mask_cropped{ext}")
    ok = cv2.imwrite(mask_cropped_path, mask_cropped)
    if not ok:
        raise IOError(f"Failed to write mask cropped image: {mask_cropped_path}")

    return mask_binary_path, masked_image_path, mask_cropped_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run safety_vlm() on a single image + bbox.json")
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help="Path to a single frame image (png/jpg). If omitted, uses DEFAULT_IMAGE_PATH",
    )
    parser.add_argument(
        "--bbox",
        default=DEFAULT_BBOX_JSON_PATH,
        help="Path to bbox json (contains {'bbox':{x,y,width,height}}). If omitted, uses DEFAULT_BBOX_JSON_PATH",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to safety_vlm()",
    )
    parser.add_argument(
        "--draw",
        dest="draw",
        action="store_true",
        default=DEFAULT_DRAW,
        help="Draw bbox on image and send the boxed image to safety_vlm() (default: enabled)",
    )
    parser.add_argument(
        "--no-draw",
        dest="draw",
        action="store_false",
        help="Disable drawing bbox and send the original image to safety_vlm()",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for boxed image (only used with --draw). Default: <image>_bounding_box_tmp.png",
    )
    parser.add_argument(
        "--temp-dir",
        default=os.path.join(os.path.dirname(__file__), "temp"),
        help="Directory to save the cropped middle-third image (default: safety_system/temp)",
    )
    parser.add_argument(
        "--sam-image",
        default=SAM_IMAGE_PATH,
        help="Path to SAM input image (png/jpg). Default: SAM_IMAGE_PATH",
    )
    parser.add_argument(
        "--sam-mask",
        default=SAM_MASK_PATH,
        help="Path to SAM mask image (png). Default: SAM_MASK_PATH",
    )
    parser.add_argument(
        "--masked-out",
        default=None,
        help="Output path for fused(masked) image when using --sam-mask. Default: <image>_masked.png in mask dir.",
    )

    args = parser.parse_args()
    # Use repo-local temp directories by default (can be overridden via --temp-dir)
    args.temp_dir = args.temp_dir
    args.temp_dir2 = os.path.join(os.path.dirname(__file__), "temp2")
    if not args.image or not args.bbox:
        raise SystemExit(
            "Please pass --image/--bbox (or set DEFAULT_IMAGE_PATH/DEFAULT_BBOX_JSON_PATH in the script)."
        )


    bbox = _load_bbox(args.bbox)

    image_to_send = args.image
    sam_image_to_send = None
    if args.draw:
        out_path = args.out
        if out_path is None:
            root, ext = os.path.splitext(args.image)
            out_path = f"{root}_bounding_box_tmp{ext or '.png'}"
        image_to_send = _draw_bbox(args.image, bbox, out_path)
        print(f"[saved] boxed image: {image_to_send}")

        # The boxed image can be large; keep the middle third before sending to the VLM.
        image_to_send = _crop_middle_third_vertical(image_to_send, args.temp_dir)
        print(f"[saved] cropped image: {image_to_send}")

    if args.sam_mask:
        if not args.sam_image:
            raise SystemExit("已提供 --sam-mask，但缺少 --sam-image")
        mask_binary_path, masked_image_path, mask_cropped_path = _save_mask_and_masked_image(
            args.sam_image,
            args.sam_mask,
            out_dir=os.path.dirname(args.sam_mask),
            masked_out_path=args.masked_out,
        )
        
        print(f"[saved] mask binary image: {mask_binary_path}")
        print(f"[saved] masked image: {masked_image_path}")
        masked_middle_third_path = _crop_middle_third_vertical(
            masked_image_path, os.path.dirname(masked_image_path) or args.temp_dir
        )
        print(f"[saved] masked middle-third image: {masked_middle_third_path}")
        print(f"[saved] mask cropped image: {mask_cropped_path}")
        sam_image_to_send = masked_middle_third_path

    # Also crop the bbox region as `image_to_send_2`.
    image_to_send_2 = _crop_bbox_region(args.image, bbox, args.temp_dir2)
    print(f"[saved] bbox cropped image: {image_to_send_2}")

    # Call API

    

    print("======default prompt:======")
    # default prompt
    # args.prompt = "Describe what you see in the frame; decide whether it is safe to continue the landing."
    # result = safety_vlm(image_to_send,args.prompt)
    # print(result)

    print("======Safety Distance======")
    # Safety distance
    # args.prompt = "What objects are in front of the aircraft? Is there a safe separation distance?"
    # result = safety_vlm(image_to_send,args.prompt)
    # print(result)


    print("======Helipad Clearance======")
    # Helipad Clearance
    # args.prompt = "Is the helipad area and the airspace directly above it clear of obstacles?"
    # result = safety_vlm(image_to_send_2,args.prompt)
    # print(result)

    print("======SAM:======")
    # SAM
    args.prompt = "Is there a helipad visible? If yes, does the image contain only the helipad? If not, describe the scene and whether it is safe to continue the landing."
    result = safety_vlm(sam_image_to_send or image_to_send, args.prompt)
    print(result)

    print("======Tracking result:======")
    # Tracking results
    # args.prompt = "Identify the object in the red box; is the box severely offset from the helipad? If there is no box, describe the scene and decide whether it is safe to continue the landing."
    # result = safety_vlm(image_to_send,args.prompt)
    # print(result)

    print("======Heading alignment:======")
    # Heading alignment
    # args.prompt = "Is the aircraft currently aligned with the helipad? Explain briefly."
    # result = safety_vlm(image_to_send,args.prompt)
    # print(result)

    print("======Environment and explanation of VLM:======")
    # Environment and explanation of VLM
    # args.prompt = "Describe the scene and justify whether it is safe to continue the landing."
    # result = safety_vlm(image_to_send,args.prompt)
    # print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
