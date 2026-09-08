"""Prompt definitions and schemas for VLM door auto-annotation.
Supports both Pydantic (if installed) and standard library dataclasses for zero-dependency execution.
"""

from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda default=None, default_factory=None, description="": default if default is not None else (default_factory() if default_factory else None)


if HAS_PYDANTIC:
    class DetectedDoor(BaseModel):
        box_2d: List[int] = Field(
            description="Bounding box in normalized coordinates [ymin, xmin, ymax, xmax] on a scale of 0 to 1000."
        )
        label: str = Field(
            description="Door state classification. Must be strictly 'open_door' or 'closed_door'."
        )
        openness_percentage: int = Field(
            description="Estimated percentage of openness (0 = fully closed, 25 = slightly ajar, 50 = half open, 100 = wide open)."
        )
        door_type: Optional[str] = Field(
            default="standard",
            description="Type of door observed, e.g. 'hinged_wood', 'glass', 'sliding', 'metal_fire_door'."
        )
        confidence: float = Field(
            default=0.9,
            description="Confidence level in the detection between 0.0 and 1.0."
        )
        reasoning: Optional[str] = Field(
            default="",
            description="Concise description of visual cues justifying the openness percentage and state."
        )

    class ImageAnnotationResponse(BaseModel):
        doors: List[DetectedDoor] = Field(
            default_factory=list,
            description="List of all detected doors in the image."
        )

else:
    from dataclasses import dataclass, field

    @dataclass
    class DetectedDoor:
        box_2d: List[int]
        label: str
        openness_percentage: int
        door_type: Optional[str] = "standard"
        confidence: float = 0.9
        reasoning: Optional[str] = ""

    @dataclass
    class ImageAnnotationResponse:
        doors: List[DetectedDoor] = field(default_factory=list)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]) -> "ImageAnnotationResponse":
            raw_doors = data.get("doors", [])
            parsed_doors = []
            for d in raw_doors:
                parsed_doors.append(
                    DetectedDoor(
                        box_2d=d.get("box_2d", [0, 0, 0, 0]),
                        label=d.get("label", "closed_door"),
                        openness_percentage=int(d.get("openness_percentage", 0)),
                        door_type=d.get("door_type", "standard"),
                        confidence=float(d.get("confidence", 0.9)),
                        reasoning=d.get("reasoning", "")
                    )
                )
            return cls(doors=parsed_doors)


DOOR_DETECTION_SYSTEM_INSTRUCTION = """
You are an expert Computer Vision annotator and Spatial Perception specialist for assistive navigation systems.
Your task is to identify every door in the provided image and generate precise bounding boxes and semantic attributes.

Guidelines for Annotation:
1. BOUNDING BOX:
   - Provide tight 2D bounding boxes in normalized coordinates [ymin, xmin, ymax, xmax] on a scale of 0 to 1000.
   - For an OPEN door: Enclose the door opening / passage AND the door leaf itself as a single functional door unit.
   - For a CLOSED door: Enclose the door leaf and its immediate frame.

2. CLASSIFICATION ('label'):
   - 'open_door': The door is ajar, swung open, sliding open, or provides an accessible passage through the wall.
   - 'closed_door': The door is latched, flush with the frame, or completely obstructing the doorway.

3. OPENNESS PERCENTAGE ('openness_percentage'):
   - Estimate the physical aperture as an integer percentage from 0 to 100:
     * 0%: Completely closed, no visible gap or passage.
     * 10% - 30%: Slightly ajar / cracked open (door unlatched, small gap visible).
     * 35% - 65%: Halfway open (partial passage).
     * 70% - 100%: Wide open (clear passage through).

4. PRECISION:
   - If no doors are present in the image, return an empty list of doors.
   - Avoid hallucinating windows or picture frames as doors.
   - Always return valid JSON adhering strictly to the schema.
"""
