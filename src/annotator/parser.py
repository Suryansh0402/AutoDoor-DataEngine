"""Coordinate conversion from VLM [ymin, xmin, ymax, xmax] to YOLO [class_id, x_center, y_center, w, h]."""

from typing import List, Tuple, Dict, Any
from .prompt_templates import DetectedDoor, ImageAnnotationResponse


DEFAULT_CLASS_MAP_2 = {
    "open_door": 0,
    "closed_door": 1,
}

GRANULAR_CLASS_MAP_3 = {
    "closed_door": 0,
    "partially_open_door": 1,
    "wide_open_door": 2,
}


def vlm_box_to_yolo(
    box_2d: List[int],
    img_width: int = 1000,
    img_height: int = 1000
) -> Tuple[float, float, float, float]:
    """
    Converts VLM coordinates [ymin, xmin, ymax, xmax] (normalized 0 to 1000)
    to YOLO format (x_center, y_center, width, height) normalized to [0.0, 1.0].
    """
    ymin, xmin, ymax, xmax = box_2d

    # Clamp bounds to [0, 1000]
    xmin = max(0, min(1000, xmin))
    xmax = max(0, min(1000, xmax))
    ymin = max(0, min(1000, ymin))
    ymax = max(0, min(1000, ymax))

    # Calculate center and dimensions in normalized [0.0, 1.0]
    box_w = (xmax - xmin) / 1000.0
    box_h = (ymax - ymin) / 1000.0
    x_center = (xmin + xmax) / 2.0 / 1000.0
    y_center = (ymin + ymax) / 2.0 / 1000.0

    return (
        round(x_center, 6),
        round(y_center, 6),
        round(box_w, 6),
        round(box_h, 6)
    )


def resolve_class_id(door: DetectedDoor, mode: str = "2-class") -> Tuple[int, str]:
    """
    Maps a detected door and its openness percentage to a YOLO class ID.
    
    Modes:
      - '2-class': 0 = open_door, 1 = closed_door (compatible with VisionWalk-Assist)
      - '3-class': 0 = closed, 1 = partially_open, 2 = wide_open
    """
    if mode == "3-class":
        pct = door.openness_percentage
        if pct <= 10:
            return GRANULAR_CLASS_MAP_3["closed_door"], "closed_door"
        elif pct <= 60:
            return GRANULAR_CLASS_MAP_3["partially_open_door"], "partially_open_door"
        else:
            return GRANULAR_CLASS_MAP_3["wide_open_door"], "wide_open_door"

    # Default 2-class mapping
    label = door.label.lower().strip()
    if "open" in label:
        return DEFAULT_CLASS_MAP_2["open_door"], "open_door"
    return DEFAULT_CLASS_MAP_2["closed_door"], "closed_door"


def annotation_to_yolo_lines(
    annotation: ImageAnnotationResponse,
    mode: str = "2-class"
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Converts ImageAnnotationResponse to lines for a YOLO .txt file and companion metadata.
    """
    yolo_lines = []
    metadata_entries = []

    for door in annotation.doors:
        class_id, resolved_label = resolve_class_id(door, mode=mode)
        xc, yc, w, h = vlm_box_to_yolo(door.box_2d)

        # Ignore degenerate boxes with zero width or height
        if w <= 0.001 or h <= 0.001:
            continue

        line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(line)

        metadata_entries.append({
            "class_id": class_id,
            "resolved_label": resolved_label,
            "original_label": door.label,
            "openness_percentage": door.openness_percentage,
            "door_type": door.door_type,
            "confidence": door.confidence,
            "reasoning": door.reasoning,
            "yolo_bbox": [xc, yc, w, h],
            "vlm_box_2d": door.box_2d
        })

    return yolo_lines, metadata_entries
