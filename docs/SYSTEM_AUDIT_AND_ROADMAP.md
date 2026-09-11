# VisionForge-AI: System Audit & Multi-Phase Engineering Roadmap 📋🚀

## 1. Executive Summary & Context

**VisionForge-AI** serves as the automated visual data collection, visual-language grounding, and dataset engineering engine for indoor door detection and spatial navigation (specifically powering the downstream assistive perception system **VisionWalk-Assist**).

This document captures:
1. The **initial system audit (Phase 0)** covering existing detector architectures, dataset status, training/inference pipelines, depth estimation, and identified technical debt.
2. The **end-to-end multi-phase architecture** transforming the project into a problem-driven ML system: VLM auto-annotation, quality control, semi-supervised teacher–student learning, hard-example mining, knowledge distillation, and metric spatial perception.

---

## 2. Phase 0 Audit: Current Repository State

### 2.1 Existing YOLO Architecture
* **Framework**: Ultralytics YOLOv8 (`ultralytics >= 8.3.x`).
* **Base Architecture**: `yolov8m.pt` (YOLOv8 Medium).
* **Input Dimensions**: $640 \times 640$ pixels (`imgsz=640`).
* **Inference Loop**: OpenCV-based real-time video acquisition loop running with confidence threshold $\tau = 0.50$.
* **Training Hyperparameters**: 200 epochs, batch size 8, transfer learning initialization from official pretrained weights.

### 2.2 Current Dataset Structure & Size
* **Local Repositories**: Currently contain **no local versioned datasets or offline images** (only directory `.gitkeep` placeholders in `data/raw`, `data/processed`, `data/output_yolo`).
* **External Dependency**: Training was historically coupled to an external Roboflow API project (`suryansh-stlog/door-detection-and-alerts/1`).
* **Audit Finding**: The project lacks a local, traceable, version-controlled dataset with persistent provenance metadata and physical scene tracking.

### 2.3 Existing Classes & Label Schema
* **VisionWalk-Assist**: Evaluates 2 classes:
  * `open` / `open_door`: Clear passages, open door leaves, unobstructed doorways.
  * `closed` / `closed_door`: Latched doors, solid barriers.
* **VisionForge-AI**: Configured for 2 modes:
  * `2-class`: `0: open_door`, `1: closed_door`.
  * `3-class`: `0: closed_door`, `1: partially_open_door`, `2: wide_open_door` (derived from estimated openness percentage).
* **Audit Finding & Class Strategy**:
  * For visual grounding and initial data engine scaling, forcing an open-vocabulary model to simultaneously localize doors *and* determine subtle physical aperture leads to high label noise.
  * **Strategy**: V1 of the expanded auto-annotation pipeline will focus strictly on **Class 0: `door`** to maximize recall and spatial precision. Aperture/openness classification will be layered downstream or handled by a dedicated classifier.

### 2.4 Training & Inference Pipelines
* **Training Pipeline**: Script-driven (`train_model.py`) via `model.train(data=..., epochs=200, imgsz=640, batch=8)`.
  * Missing: Seed locking, experiment logging (TensorBoard/W&B/JSON tracking), custom data augmentations, stratified cross-validation.
* **Inference Pipeline**: Real-time webcam / RTSP IP camera feed (`main_Webcam.py`, `main_IP_Cam.py`).
  * Captures frames at 30 FPS $\rightarrow$ MiDaS depth inference $\rightarrow$ YOLOv8 forward pass $\rightarrow$ Centroid distance calculation $\rightarrow$ Non-blocking TTS alert playback.

### 2.5 Evaluation Metrics
* **Current State**: No benchmark metrics (mAP50, mAP50-95, precision, recall, confusion matrix, or latency benchmarks) are stored or tracked locally.
* **Requirement**: Establish a gold-standard, human-verified validation/test benchmark before evaluating data expansion gains.

### 2.6 Depth & Distance Implementation
* **Model**: Intel ISL `MiDaS_small` via PyTorch Hub.
* **Prediction**: Bicubic upsampled relative inverse disparity map $\mathcal{D}(y, x)$.
* **Distance Formulation**: Single-pixel centroid query:
  $$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}, \quad Z = \frac{k_{\text{depth}}}{\mathcal{D}(c_y, c_x)}$$
  where $k_{\text{depth}} = 400$ (laptop webcam heuristic).
* **Identified Flaws**:
  1. **Single-pixel fragility**: Centroid lookup fails on open doors because the center of an open doorway views the background corridor/room, registering a false distant depth.
  2. **Relative vs. Metric**: MiDaS is affine-invariant relative depth; without metric camera calibration or metric-depth models (e.g. ZoeDepth/Metric3D), absolute meter distance is inaccurate.
  3. **Direction / Angle**: Lateral bearing logic ($\arctan((c_x - c_0) / f_x)$) is described in architecture specs but omitted in runtime code.

### 2.7 Reusable Code Assets
* `src/collector/frame_extractor.py`: Laplacian variance blur rejection (`is_frame_blurry`) and aspect-ratio downscaling (`resize_frame_maintaining_aspect`).
* `src/annotator/parser.py`: Normalized coordinate conversion from $[0, 1000]$ to standard YOLO floating-point center format with bounds clamping.
* `src/dataset/visualizer.py`: OpenCV annotation overlays with bounding boxes and status labels.
* `tests/test_pipeline.py`: Unit test coverage for coordinate math and class resolution.
* `VoiceAlert` in `VisionWalk-Assist`: Thread-safe, non-blocking asynchronous TTS engine with anti-stacking `is_busy` locks.

### 2.8 Technical Debt & Blockers
* ⚠️ **Video Train/Val Leakage**: `exporter.py` currently performs a random shuffle across all individual frames. Consecutive frames from the same video walkthrough will leak across train and validation splits, artificially inflating evaluation metrics.
* ⚠️ **Exporter Bug**: `exporter.py` calls `random.seed()` and `random.shuffle()` without importing Python's `random` module.
* ⚠️ **VLM Coupling**: Annotation logic is tightly coupled to Google Gemini API SDK; needs a generic `BaseGroundingModel` interface to support open-vocabulary detectors (Grounding DINO, Florence-2, OWLv2, local VLMs).
* ⚠️ **Quality Filtering Missing**: Generated annotations currently lack confidence thresholding, IoU deduplication, and a human review queue interface.

---

## 3. End-to-End System Architecture

```
STAGE A: DATA ENGINE
Unlabelled Indoor Images / Video Footage
       │
       ▼
Ingestion & Scene/Video Source Tagging
       │
       ▼
Laplacian Blur Filter + Near-Duplicate Rejection
       │
       ▼
Open-Vocabulary / Grounding VLM (Prompt: "door")
       │
       ▼
Confidence & Quality Filtering
       ├── High Confidence (≥ 0.90) ──► Automatic Acceptance (with spot checks)
       ├── Uncertain (0.50 - 0.89)   ──► Human Review Queue
       └── Low Confidence (< 0.50)   ──► Hard-Example Queue / Discarded
       │
       ▼
Group-Aware YOLO Dataset Splitter (Zero-Leakage by Building/Video)
       │
       ▼
Versioned Dataset Repository (v1.0, v2.0...)

STAGE B: BASELINE
Verified Dataset v1.0 ──► Train Baseline YOLOv8m ──► Benchmark mAP50 / mAP50-95

STAGE C: SEMI-SUPERVISED TEACHER–STUDENT LEARNING
Labelled Data + Unlabelled Pool
       │
       ▼
Teacher Detector (EMA Updated) ──► High-Confidence Pseudo-Labels
       │                                       │
       └──────────────┬────────────────────────┘
                      ▼
               Student YOLOv8
                      │ (Gradient updates)
                      ▼
               Update Teacher Weights: θ_teacher = α·θ_teacher + (1-α)·θ_student

STAGE D: HARD-EXAMPLE MINING & ACTIVE LEARNING
Run Detector over unlabelled pool ──► Flag high-disagreement / low-confidence / hard negatives
       │
       ▼
Human-in-the-Loop Review Queue ──► High-Value Training Pairs ──► Iterative Retraining

STAGE E: KNOWLEDGE DISTILLATION (DEPLOYMENT)
Strong Teacher (YOLOv8m/x) ──► Distillation Loss ──► Lightweight Student (YOLOv8n/s)
Benchmark: Accuracy (mAP) vs. Edge Latency (FPS/ms)

STAGE F: PERCEPTION & SPATIAL GEOMETRY
Camera Frame ──► YOLO Door Detector + Metric Depth Estimator
       │
       ▼
Trimmed Median Depth over Door Region + Calibrated Camera Intrinsics
       │
       ▼
Output: { door_detected: true, distance_m: 2.4, angle_deg: -12.5, direction: "left" }
```

---

## 4. Multi-Phase Execution Roadmap

| Phase | Title | Core Objective | Key Deliverables |
| :---: | :--- | :--- | :--- |
| **Phase 0** | **System Audit & Baseline Fixes** | Inspect current architecture, fix technical debt, formalize schemas. | Audit doc, fix `exporter.py` import bug, design zero-leakage group splitter. |
| **Phase 1** | **Auto-Annotation POC** | Grounding pipeline detecting "door" and exporting valid YOLO labels. | `BaseGroundingModel` interface, Gemini/Grounding-DINO adapter, CLI annotation tool, visual preview overlays. |
| **Phase 2** | **Quality Control & Review Queue** | Confidence gating and rapid human-in-the-loop verification. | Review queue schema, acceptance policies ($\tau_{\text{high}}, \tau_{\text{low}}$), minimal verification CLI/UI tool. |
| **Phase 3** | **Data Engine Scaling & Anti-Leakage** | Batch video extraction, deduplication, safe group-based train/val/test splits. | Video scene sampler, perceptual hashing deduplication, source-grouped dataset exporter. |
| **Phase 4** | **Baseline Retraining** | Train baseline detector on verified dataset; establish ground-truth metrics. | Reproducible training script, validation benchmark suite, evaluation report (mAP50, mAP50-95). |
| **Phase 5** | **Semi-Supervised Teacher–Student SSL** | Exploit unlabelled indoor data using pseudo-labeling with EMA teacher updates. | Teacher–student training loop, confidence filtering, augmentation consistency, SSL ablation study. |
| **Phase 6** | **Hard-Example Mining & Active Learning** | Mine failure modes (distant doors, occlusions, hard negatives like cabinets/mirrors). | Disagreement mining script, active learning review cycle, targeted robustness evaluation. |
| **Phase 7** | **Knowledge Distillation** | Compress strong teacher into lightweight real-time model (YOLOv8s/n). | Distillation loss module, student benchmark comparison (mAP vs FPS vs Model Size). |
| **Phase 8** | **Final Perception & Camera Geometry** | Robust depth extraction (trimmed median) + camera intrinsics for bearing/distance. | Spatial perception module, calibrated angle/distance output, live edge demo. |

---

## 5. Non-Negotiable Engineering Rules

1. **Zero Data Leakage Across Splits**: Video frames or images from the same physical room, hallway, or video capture must never be split across train, val, and test. All splits must be grouped by **source entity** (Video ID or Building/Room ID).
2. **Never Blindly Trust Auto-Labels**: All auto-generated labels must maintain provenance tags (`model_name`, `confidence`, `review_status`, `timestamp`). No pseudo-label is added to ground truth without passing strict confidence gates or human review.
3. **Reproducible Experiment Tracking**: Every experiment records architecture, dataset hash, split configuration, optimizer parameters, learning rate, random seed, and hardware environment.
4. **Metric Honesty**: Every architectural addition (VLM expansion, SSL, distillation) must be accompanied by empirical ablation metrics.
