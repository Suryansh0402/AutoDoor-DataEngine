"""Unified Command-Line Interface for AutoDoor-DataEngine."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from src.collector.frame_extractor import extract_frames_from_video
from src.collector.image_loader import preprocess_image_batch
from src.annotator.vlm_engine import VLMEngine
from src.annotator.parser import annotation_to_yolo_lines
from src.dataset.visualizer import draw_annotations_on_image
from src.dataset.exporter import export_yolo_dataset


def cmd_collect(args):
    """Collects and optimizes frames from video or raw images."""
    if args.video:
        print(f"🎬 Extracting frames from video: {args.video}")
        extract_frames_from_video(
            video_path=args.video,
            output_dir=args.output,
            interval_seconds=args.interval,
            max_dim=args.max_dim,
            skip_blurry=not args.no_blur_filter
        )
    elif args.images:
        print(f"📁 Processing raw images from: {args.images}")
        preprocess_image_batch(
            input_dir=args.images,
            output_dir=args.output,
            max_dim=args.max_dim,
            skip_blurry=not args.no_blur_filter
        )
    else:
        print("❌ Error: Specify either --video <path> or --images <dir>.")
        sys.exit(1)


def cmd_annotate(args):
    """Queries the VLM to generate bounding boxes and openness percentages."""
    input_dir = Path(args.input)
    labels_dir = Path(args.labels_dir)
    metadata_dir = Path(args.metadata_dir)
    preview_dir = Path(args.preview_dir) if args.generate_previews else None

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    if not image_files:
        print(f"❌ No images found in '{input_dir}'. Run collect first!")
        sys.exit(1)

    print(f"🤖 Starting VLM auto-annotation on {len(image_files)} images using {args.model}...")
    engine = VLMEngine(api_key=args.api_key, model_name=args.model)

    annotated_count = 0
    total_doors = 0

    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Annotating '{img_file.name}'...")
        try:
            annotation = engine.annotate_image(str(img_file))
            yolo_lines, metadata = annotation_to_yolo_lines(annotation, mode=args.mode)

            # Save YOLO .txt
            txt_path = labels_dir / f"{img_file.stem}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            # Save JSON metadata (openness pct, reasoning, etc.)
            meta_path = metadata_dir / f"{img_file.stem}.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "image": img_file.name,
                    "doors": metadata
                }, f, indent=2)

            # Render preview image if requested
            if preview_dir and metadata:
                preview_path = preview_dir / f"preview_{img_file.name}"
                draw_annotations_on_image(str(img_file), metadata, str(preview_path))

            annotated_count += 1
            total_doors += len(metadata)

        except Exception as e:
            print(f"⚠️ Failed to annotate '{img_file.name}': {e}")

    print(f"\n🎉 Annotation finished!")
    print(f"   ├─ Images processed: {annotated_count}")
    print(f"   ├─ Total doors found: {total_doors}")
    print(f"   ├─ YOLO labels saved: {labels_dir}")
    print(f"   └─ Metadata saved:    {metadata_dir}")


def cmd_export(args):
    """Exports dataset into train/val split with data.yaml."""
    export_yolo_dataset(
        processed_images_dir=args.images,
        labels_dir=args.labels,
        output_yolo_dir=args.output,
        val_split=args.val_split,
        mode=args.mode
    )


def main():
    parser = argparse.ArgumentParser(
        description="AutoDoor-DataEngine: Automated Visual Data Collection & VLM-Powered Annotation Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: collect
    p_collect = subparsers.add_parser("collect", help="Extract and downscale frames from video or photo directory")
    p_collect.add_argument("--video", type=str, help="Path to input video file")
    p_collect.add_argument("--images", type=str, help="Directory containing raw images")
    p_collect.add_argument("--output", type=str, default="data/processed", help="Output directory for processed images")
    p_collect.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between frames")
    p_collect.add_argument("--max-dim", type=int, default=640, help="Max dimension for token optimization")
    p_collect.add_argument("--no-blur-filter", action="store_true", help="Disable Laplacian blur filtering")

    # Command: annotate
    p_anno = subparsers.add_parser("annotate", help="Run VLM auto-annotation on processed images")
    p_anno.add_argument("--input", type=str, default="data/processed", help="Directory of processed images")
    p_anno.add_argument("--labels-dir", type=str, default="data/labels", help="Directory to save YOLO .txt labels")
    p_anno.add_argument("--metadata-dir", type=str, default="data/metadata", help="Directory to save rich JSON metadata")
    p_anno.add_argument("--preview-dir", type=str, default="output_preview", help="Directory for visual verification images")
    p_anno.add_argument("--generate-previews", action="store_true", default=True, help="Draw bounding boxes on preview images")
    p_anno.add_argument("--model", type=str, default="gemini-2.0-flash", help="VLM model identifier")
    p_anno.add_argument("--mode", type=str, choices=["2-class", "3-class"], default="2-class", help="Class resolution mode")
    p_anno.add_argument("--api-key", type=str, default=None, help="Gemini API Key (optional, defaults to env var)")

    # Command: export
    p_export = subparsers.add_parser("export", help="Split dataset into train/val and generate data.yaml")
    p_export.add_argument("--images", type=str, default="data/processed", help="Processed images directory")
    p_export.add_argument("--labels", type=str, default="data/labels", help="Labels directory")
    p_export.add_argument("--output", type=str, default="data/output_yolo", help="YOLO training root directory")
    p_export.add_argument("--val-split", type=float, default=0.2, help="Validation set percentage (e.g. 0.2 = 20%)")
    p_export.add_argument("--mode", type=str, choices=["2-class", "3-class"], default="2-class", help="Class mode for data.yaml")

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
