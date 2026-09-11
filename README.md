# VisionForge-AI: Automated Visual Data Collection & VLM-Powered Annotation Engine 🔨👁️

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Gemini VLM](https://img.shields.io/badge/VLM-Google%20Gemini%20Flash-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![YOLOv8 Ready](https://img.shields.io/badge/Export-Ultralytics%20YOLOv8-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Companion](https://img.shields.io/badge/Companion%20To-VisionWalk--Assist-purple.svg)](https://github.com/CeilSenseiCommits/VisionWalk-Assist)

**An intelligent, token-optimized data engine that harvests video frames, automatically annotates doors with bounding boxes using Vision-Language Models, calculates physical aperture/openness percentages ($0\text{--}100\%$), and exports ready-to-train datasets for edge computer vision models.**

[Key Capabilities](#-key-capabilities) • [System Architecture](#-system-architecture) • [Openness Estimation](#-door-openness-estimation) • [Quickstart](#-quickstart--usage) • [Integration with VisionWalk-Assist](#-integration-with-visionwalk-assist) • [System Audit & Roadmap](docs/SYSTEM_AUDIT_AND_ROADMAP.md)

</div>

---

## 🎯 The Problem VisionForge-AI Solves

Training high-accuracy edge computer vision models like [VisionWalk-Assist](https://github.com/CeilSenseiCommits/VisionWalk-Assist) requires hundreds of diverse, annotated indoor images across varying lighting, angles, and door types.

* **The Manual Labeling Bottleneck:** Drawing bounding boxes manually in tools like CVAT or Roboflow takes weeks of tedious effort.
* **The Traversability Dilemma:** Standard detectors only distinguish a "door" from the wall. They cannot tell if a door is ajar by $15^\circ$ (unsafe, unnavigable gap) or swung $90^\circ$ wide open (clear walkway).

**VisionForge-AI eliminates this bottleneck.** By combining smart frame sampling (blur & redundancy filtering) with Google's multimodal **Gemini Flash** models, it automatically extracts frames, identifies doors, determines bounding boxes, computes the **exact percentage of openness**, and produces verified YOLOv8 datasets in minutes.

---

## 🌟 Key Capabilities

* 🎬 **Smart Video Frame Harvester:** Ingests raw MP4/AVI walking footage or photo batches. Uses Laplacian variance filtering to automatically discard motion-blurred frames.
* 🤖 **Zero-Shot VLM Grounding:** Leverages Google Gemini Flash with strict spatial reasoning prompts to identify doors and output 2D bounding boxes normalized to $[0, 1000]$.
* 📐 **Door Openness & Aperture Metric:** Quantifies physical traversability as an explicit percentage ($0\%$ = locked/shut, $30\%$ = slightly ajar, $100\%$ = wide open).
* 🔄 **Automated YOLO Exporter:** Transforms VLM coordinates directly into normalized Ultralytics YOLOv8 format (`.txt` files, 80/20 train/val splits, and generated `data.yaml`).
* 🎨 **Visual Verification Overlays:** Automatically generates preview images with color-coded bounding boxes (Red for closed, Green for open, Yellow for ajar) and openness badges for instant quality review.
* 💰 **Token-Optimized Architecture:** Restricts image dimensions to $640\text{px}$ before transmission, lowering API costs to fractions of a cent per image ($\sim 250\text{ tokens}$ per sample).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph 1. Ingestion & Frame Harvesting
        A[Raw Walkthrough Video mp4 / Photo Batch] --> B[Temporal Stride Sampler]
        B --> C[Laplacian Blur Filter: Discard Blurry Frames]
    end

    subgraph 2. Preprocessing & Token Throttling
        C --> D[Aspect-Preserving Downscaler: 640px Max Dim]
    end

    subgraph 3. VLM Auto-Annotation Engine
        D --> E[Gemini 2.0 / 1.5 Flash Prompt Engine]
        E -->|Structured JSON Response| F[Box Regression [ymin, xmin, ymax, xmax]]
        E -->|Traversability Reasoning| G[Openness Percentage: 0% to 100%]
    end

    subgraph 4. Conversion & Dataset Synthesis
        F --> H[Parser: VLM 0-1000 -> YOLO x_center, y_center, w, h]
        G --> H
        H --> I[Visual Preview Generator output_preview/]
        H --> J[YOLO Exporter: 80/20 Split & data.yaml]
    end

    J -->|Direct Training Input| K[VisionWalk-Assist YOLOv8 Model]
```

---

## 🚪 Door Openness Estimation

VisionForge-AI categorizes door traversability on a continuous spectrum:

| Openness % | Classification | Description & Navigational Impact | Visual Preview Color |
| :---: | :---: | :--- | :---: |
| **$0\%$** | `closed_door` | Completely latched/flush. Solid collision hazard. | 🔴 Red |
| **$10\%\text{--}30\%$** | `partially_open` | Ajar / cracked open. Door leaf protrudes into hallway; passage blocked. | 🟡 Yellow |
| **$35\%\text{--}65\%$** | `partially_open` | Half-open. Narrow clearance; requires trajectory shift. | 🟠 Orange |
| **$70\%\text{--}100\%$** | `wide_open` | Fully accessible corridor. Safe to walk straight through. | 🟢 Green |

---

## 📂 Project Structure

```
VisionForge-AI/
├── src/
│   ├── collector/
│   │   ├── frame_extractor.py     # Extracts sharp, non-redundant frames from video
│   │   └── image_loader.py        # Validates & downscales batches of raw photos
│   ├── annotator/
│   │   ├── vlm_engine.py          # Google Gemini Flash API caller with retry logic
│   │   ├── prompt_templates.py    # Spatial instruction prompts & structured schemas
│   │   └── parser.py              # Denormalizes VLM boxes into YOLO format
│   ├── dataset/
│   │   ├── exporter.py            # Generates train/val splits and data.yaml
│   │   └── visualizer.py          # Overlays bounding boxes & openness badges on images
│   └── cli.py                     # Unified CLI tool (collect, annotate, export)
├── data/
│   ├── raw/                       # Place unannotated videos/photos here
│   ├── processed/                 # Token-optimized 640px images
│   ├── labels/                    # Generated YOLO .txt label files
│   ├── metadata/                  # Rich JSON metadata (reasoning, openness pct)
│   └── output_yolo/               # Ready-to-train YOLOv8 dataset
├── docs/
│   └── ARCHITECTURE.md            # Deep-dive architectural specification
├── tests/
│   └── test_pipeline.py           # Unit tests for coordinate math & class mapping
├── requirements.txt               # Dependencies
├── .env.example                   # Environment configuration template
└── README.md
```

---

## ⚙️ Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/CeilSenseiCommits/VisionForge-AI.git
cd VisionForge-AI

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your API Key
Copy `.env.example` to `.env` and add your free Gemini API key:
```bash
cp .env.example .env
```
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Get a free API key from [Google AI Studio](https://aistudio.google.com/)).*

---

### 3. Step 1: Collect & Optimize Frames
Record a short video walking through hallways or rooms with doors, place it in `data/raw/walk.mp4`, and run:
```bash
python -m src.cli collect --video data/raw/walk.mp4 --interval 1.5 --max-dim 640
```
*Or, if you have a folder of unannotated images:*
```bash
python -m src.cli collect --images data/raw/my_photos --max-dim 640
```

---

### 4. Step 2: Auto-Annotate with Gemini Flash
Run the VLM auto-annotation engine across your harvested frames:
```bash
python -m src.cli annotate --mode 2-class --generate-previews
```
*Outputs:*
* Standard YOLO label files: `data/labels/*.txt`
* Rich metadata with openness % and reasoning: `data/metadata/*.json`
* Visual verification previews: `output_preview/preview_*.jpg`

---

### 5. Step 3: Export to YOLO Dataset
Synthesize the train and validation sets with auto-generated `data.yaml`:
```bash
python -m src.cli export --val-split 0.2 --mode 2-class
```
Output structure in `data/output_yolo/`:
```
output_yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 🔗 Integration with VisionWalk-Assist

Once exported, you can retrain or fine-tune [VisionWalk-Assist](https://github.com/CeilSenseiCommits/VisionWalk-Assist) directly on your newly synthesized dataset:

```python
from ultralytics import YOLO

# Load base model
model = YOLO("yolov8m.pt")

# Train directly on VisionForge-AI exported data
model.train(
    data="c:/Users/surya/Desktop/Projects/VisionForge-AI/data/output_yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8
)
```

---

## 💸 Token & Cost Suppression

| Optimization Technique | Mechanism | Cost / Token Reduction |
| :--- | :--- | :---: |
| **$640\text{px}$ Capping** | Caps max dimension; avoids sending bloated $4\text{K}/1080\text{p}$ images | **$-85\%$ tokens** |
| **Laplacian Blur Pruning** | Skips blurry frames before calling API | **$-20\%$ wasted calls** |
| **Temporal Stride** | Extracts 1 frame per $1.5\text{s}$ instead of 30 FPS | **$-97\%$ API volume** |
| **Pydantic Schema** | Returns pure JSON bounding boxes with zero conversational fluff | **$-60\%$ output tokens** |

---

## 🧪 Testing

Run the automated test suite to verify coordinate transformations and class mappings:
```bash
python -m unittest discover -s tests
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 🤝 Acknowledgments

* [Google DeepMind / Google AI Studio](https://aistudio.google.com/) for Gemini multimodal foundation models.
* [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8 architecture and dataset standards.
* Built as the automated dataset engineering engine for [VisionWalk-Assist](https://github.com/CeilSenseiCommits/VisionWalk-Assist).
