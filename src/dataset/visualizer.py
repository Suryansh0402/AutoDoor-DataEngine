"""Visualizer for previewing auto-generated bounding boxes, classes, and openness percentages."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def draw_annotations_on_image(
    image_path: str,
    metadata_entries: List[Dict[str, Any]],
    output_preview_path: str
) -> str:
    """
    Renders bounding boxes and openness badges directly onto an image copy for review.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    for entry in metadata_entries:
        xc, yc, bw, bh = entry["yolo_bbox"]
        label = entry["resolved_label"]
        pct = entry.get("openness_percentage", 0)

        # Convert normalized YOLO back to pixel coordinates
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)

        # Clamp to image frame
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # Color coding: Green for open, Red for closed, Yellow for partially open
        if "closed" in label:
            color = (0, 0, 240)        # Red
        elif "partially" in label:
            color = (0, 200, 255)      # Yellow/Orange
        else:
            color = (0, 230, 70)       # Green

        # Draw main bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw header badge
        badge_text = f"{label.upper()} ({pct}% Open)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

        badge_y1 = max(0, y1 - text_h - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + text_w + 10)

        # Draw badge background
        cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Draw badge text in dark contrast
        cv2.putText(
            img,
            badge_text,
            (x1 + 5, badge_y2 - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    os.makedirs(os.path.dirname(output_preview_path), exist_ok=True)
    cv2.imwrite(output_preview_path, img)
    return output_preview_path
