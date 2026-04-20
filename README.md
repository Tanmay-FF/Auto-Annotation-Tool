# Auto-Annotation Tool

An end-to-end annotation and training platform for object detection. Combines **manual annotation**, **AI-assisted auto-labeling** (Claude, Qwen2.5-VL, Grounding DINO, OWL-ViT, Florence-2, YOLOE), and **one-click YOLO training** in a single desktop application.

## Features

- **Manual Annotation** — draw, resize, move bounding boxes on images. Keyboard shortcuts for fast labeling.
- **Video Support** — open video files, extract keyframes (every N frames or N uniform), annotate them.
- **YOLO Auto-Annotate** — pick any YOLO `.pt` model, map its classes to your workspace classes, auto-label entire directories or videos. Supports multi-class mapping in a single pass.
- **VLM Auto-Annotation** — 8+ vision-language detection models:
  - **Claude (API)** — understands natural-language label definitions
  - **Qwen2.5-VL 3B** — local grounded detection with context understanding
  - **Grounding DINO** (base/tiny)
  - **OWL-ViT / OWL-v2**
  - **Florence-2** (base/large)
  - **YOLOE** — custom YOLOE `.pt` files
- **Few-shot Filtering** — CLIP/SigLIP embedding similarity against your example crops
- **SAM2 Refinement** — tighten bounding boxes using segmentation masks
- **VQA Verification** — second-pass yes/no validation (Florence-2, Qwen2-VL, Qwen2.5-VL, moondream2, PaliGemma, LLaVA GGUF)
- **One-Click YOLO Training** — trains YOLOv5/6/8/11/12/26 with customizable hyperparameters (LR, warmup, weight decay, optimizer, freeze layers, patience, augmentation presets). Live log with per-epoch losses, mAP50, mAP50-95.
- **Dashboard** — visual analytics: class distribution, coverage, box count histogram, flagged images (empty / sparse / outliers).
- **Active Learning** — automatically captures user corrections as few-shot examples.
- **Resizable Panel** — drag the divider to expand/shrink the control panel.

## Screenshots

_(Add screenshots here)_

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Tanmay-FF/auto-annotation-tool.git
cd auto-annotation-tool
```

### 2. Create an environment (no admin required)

```bash
conda create -p ./envs/annotator python=3.11 -y
conda activate ./envs/annotator
pip install -r requirements.txt
```

Or with a standard venv:

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 3. Optional: install `claude-agent-sdk` for Claude support

```bash
pip install claude-agent-sdk
```

You'll need a Claude API key configured in your environment.

### 4. Optional: install `llama-cpp-python` for LLaVA GGUF support

```bash
pip install llama-cpp-python
# For GPU: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## Running

```bash
python detection_labelling.py
```

## Quick Start

1. **Load Images** — Click "Open Images" and select a folder (or "Open Video" to extract keyframes).
2. **Load Classes** — Click "Load classes.txt" to load your class definitions (one class per line).
3. **Annotate manually** — Draw boxes by clicking and dragging on the canvas.
4. **OR auto-annotate** — Switch to the VLM tab, select a model (Claude is fastest for custom concepts), add detection prompts, and click "Run VLM Annotation".
5. **Train** — Switch to the Train tab, pick a YOLO base model, set epochs, and click "Start Training".
6. **Review** — Switch to the Dashboard tab to see class distribution, coverage, and flagged images.

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` / `D` | Previous / Next image |
| `Del` | Delete selected bounding box |
| `1`–`9` | Select class by number |
| `Ctrl+S` | Save annotations |
| `Shift+click` | Force draw new box (ignores existing) |

## Project Structure

```
.
├── detection_labelling.py              # Main app (tkinter UI, canvas, manual annotation)
├── detection_labelling_auto_annotate.py # VLM models, few-shot store, VQA verifier
├── yolo_trainer.py                      # Training tab (dataset prep, callbacks, live log)
├── dashboard.py                         # Analytics dashboard (charts, metrics)
├── smart_annotator.py                   # (deprecated) detect-then-classify pipeline
├── requirements.txt                     # Python dependencies
└── DEMO_SCRIPT.md                       # Demo flow / talking points
```

## Annotation Format

All annotations are saved in **YOLO format** (`.txt` files alongside each image):

```
class_idx xc yc w h
```

Where `xc`, `yc`, `w`, `h` are normalized (0.0–1.0). Segmentation masks are saved in `*_seg.txt` with polygon coordinates.

## Model Compatibility

| Library | Version |
|---|---|
| `transformers` | 4.51.3 (recommended) |
| `torch` | ≥ 2.0 |
| `ultralytics` | ≥ 8.0 |
| `pillow` | ≥ 9.0 |

## Workflow

```
Raw Images/Video
      │
      ▼
[Manual Annotation] or [VLM Auto-Annotation (Claude/Qwen2.5-VL/YOLOE/...)]
      │
      ▼
[Review & Correct] ◀── Active Learning captures corrections
      │
      ▼
[Train YOLO Model] ──▶ best.pt
      │
      ▼
[Use trained model for Auto-Annotate on new data]
      │
      ▼
  Repeat cycle → Better model each iteration
```

## License

MIT

## Contributing

Pull requests welcome! For major changes, please open an issue first to discuss what you'd like to change.
