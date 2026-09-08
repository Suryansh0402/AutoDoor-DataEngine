"""Unit tests for VisionForge-AI parsing and coordinate transformations."""

import unittest
from src.annotator.prompt_templates import DetectedDoor, ImageAnnotationResponse
from src.annotator.parser import (
    vlm_box_to_yolo,
    resolve_class_id,
    annotation_to_yolo_lines,
    DEFAULT_CLASS_MAP_2,
    GRANULAR_CLASS_MAP_3
)


class TestVisionForgePipeline(unittest.TestCase):

    def test_vlm_box_to_yolo(self):
        # A box centered at (500, 500) spanning 200 to 800 in x, and 100 to 900 in y
        box_2d = [100, 200, 900, 800]
        xc, yc, w, h = vlm_box_to_yolo(box_2d)

        self.assertAlmostEqual(xc, 0.5, places=3)
        self.assertAlmostEqual(yc, 0.5, places=3)
        self.assertAlmostEqual(w, 0.6, places=3)
        self.assertAlmostEqual(h, 0.8, places=3)

    def test_class_resolution_2_class(self):
        open_door = DetectedDoor(
            box_2d=[100, 100, 900, 900],
            label="open_door",
            openness_percentage=85
        )
        closed_door = DetectedDoor(
            box_2d=[100, 100, 900, 900],
            label="closed_door",
            openness_percentage=0
        )

        cls_id_open, label_open = resolve_class_id(open_door, mode="2-class")
        cls_id_closed, label_closed = resolve_class_id(closed_door, mode="2-class")

        self.assertEqual(cls_id_open, 0)
        self.assertEqual(label_open, "open_door")
        self.assertEqual(cls_id_closed, 1)
        self.assertEqual(label_closed, "closed_door")

    def test_class_resolution_3_class(self):
        door_shut = DetectedDoor(box_2d=[0, 0, 1000, 1000], label="closed_door", openness_percentage=5)
        door_ajar = DetectedDoor(box_2d=[0, 0, 1000, 1000], label="open_door", openness_percentage=35)
        door_wide = DetectedDoor(box_2d=[0, 0, 1000, 1000], label="open_door", openness_percentage=90)

        c0, _ = resolve_class_id(door_shut, mode="3-class")
        c1, _ = resolve_class_id(door_ajar, mode="3-class")
        c2, _ = resolve_class_id(door_wide, mode="3-class")

        self.assertEqual(c0, 0)
        self.assertEqual(c1, 1)
        self.assertEqual(c2, 2)

    def test_annotation_to_yolo_lines(self):
        annotation = ImageAnnotationResponse(
            doors=[
                DetectedDoor(box_2d=[100, 200, 900, 800], label="open_door", openness_percentage=70),
                DetectedDoor(box_2d=[50, 50, 800, 500], label="closed_door", openness_percentage=0)
            ]
        )
        lines, meta = annotation_to_yolo_lines(annotation, mode="2-class")
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(meta), 2)
        self.assertTrue(lines[0].startswith("0 "))
        self.assertTrue(lines[1].startswith("1 "))


if __name__ == "__main__":
    unittest.main()
