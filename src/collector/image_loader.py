import os
import cv2
from pathlib import Path
from typing import List
from .frame_extractor import resize_frame_maintaining_aspect, is_frame_blurry


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_image_batch(
    input_dir: str,
    output_dir: str,
    max_dim: int = 640,
    skip_blurry: bool = False,
    blur_threshold: float = 80.0
) -> List[str]:
    """
    Processes a directory of raw images, downscaling them to max_dim to optimize VLM tokens.
    
    Args:
        input_dir: Path to directory containing raw images
        output_dir: Directory where processed, token-optimized images will be stored
        max_dim: Max bounding dimension for resizing
        skip_blurry: Flag to drop blurry images
        blur_threshold: Variance cutoff for blur filter
        
    Returns:
        List of paths to processed images.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    processed_paths = []
    print(f"📁 Found {len(image_files)} images in '{input_dir}' to process.")

    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠️ Warning: Could not read '{img_file.name}', skipping.")
            continue

        if skip_blurry and is_frame_blurry(img, blur_threshold):
            continue

        resized = resize_frame_maintaining_aspect(img, max_dim)
        out_path = os.path.join(output_dir, f"{img_file.stem}.jpg")
        cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        processed_paths.append(out_path)

    print(f"✅ Successfully processed {len(processed_paths)} images to '{output_dir}'.")
    return processed_paths
