# Auto Labelling Tool - Demo Script

## System Overview (2 min)

This is an end-to-end annotation and training platform that combines **manual annotation**, **AI-assisted auto-labeling** (multiple VLM models), and **one-click YOLO training** in a single desktop application.

**Problem it solves:** One of the biggest bottlenecks in building vision models is annotation time.
So we built a tool that reduces manual labeling by combining detection models and VLMs in one workflow.

**Four tabs:**
- **Annotate** - Manual bounding box annotation + YOLO auto-annotate
- **VLM** - AI-powered detection using Qwen2.5-VL, Grounding DINO, OWL-ViT, Florence-2, YOLOE
- **Dashboard** - Class distribution, coverage, flagged images analytics
- **Train** - One-click YOLO model training from your annotations

---

## Demo Flow

### Part 1: Manual Annotation Basics (3 min)

**Setup:**
1. Click **"Load classes.txt"** to load your class definitions (e.g., `cart`, `nested_cart`)
2. Click **"Open Images"** to load an image directory

**Show:**
- Draw a bounding box by clicking and dragging on the canvas
- Select a class from the Classes listbox (or press 1-9 keys)
- Resize a box by dragging its edges
- Move a box by dragging inside it
- Delete a box by selecting it and pressing **Del**
- Navigate between images with **A** (prev) / **D** (next)
- Annotations auto-save in YOLO format (.txt files)

**Key point:** _"Manual annotation works, but it's slow. Let's see how we can automate this."_

---

### Part 2: YOLO Auto-Annotation (3 min)

**Show two input types:**
1. Click **"Auto Annotate (YOLO)"** in the Annotate tab
2. Select your YOLO .pt model file
3. **Choose source:** show both options
   - **Image directory** - select a folder of images
   - **Video file** - select a video, configure frame extraction (every N frames or total N uniform)

**Multi-class mapping:**
4. In the mapping dialog, click **"+ Add"** multiple times to map several model classes at once
   - e.g., `0: person -> 0: person` AND `2: car -> 1: vehicle`
5. Click **"Start Annotation"**
6. Show the annotated results appearing on the canvas

**Discard and re-annotate:**
7. Click **"Discard Current"** to clear the current image
8. Or click **"Discard All"** to wipe all annotations (with confirmation)

**Key point:** _"YOLO auto-annotate is fast but limited to classes the model already knows. For custom/domain-specific objects, we need VLM models."_

---

### Part 3: VLM Auto-Annotation (5 min)

**Switch to the VLM tab.**

#### 3a: Load the VLM Model
1. Select **"Qwen2.5-VL 3B"** from the Model dropdown
2. Click **"Load Model"** - loads weights to GPU
3. Status shows the loaded model name

**Key point:** _"Qwen2.5-VL is a grounded vision-language model that runs locally on your GPU and understands natural-language label descriptions."_

#### 3b: Configure Detection Prompts
1. Type `nested_cart` in the Text field
2. Select the target class from "Map to" dropdown
3. Click **Add**
4. Repeat for `cart` mapped to its class

#### 3c: Label Context (KEY DIFFERENTIATOR)
Show the **Label context** text box below the prompts:
```
nested_cart = a shopping cart with one or more other carts stacked/nested inside it
cart = a single standalone shopping cart with NO other cart inside
```

**Key point:** _"This is what makes VLM annotation powerful. Unlike other detection models that only understand keywords, the VLM reads these definitions and applies them strictly. This dramatically reduces false positives for domain-specific concepts like 'nested cart'."_

#### 3d: Run Annotation
1. Set Scope to **"Current"** for a live demo
2. Click **"Run VLM Annotation"**
3. Show the bounding boxes appearing on the canvas
4. Navigate to verify results

**Key point:** _"The model understood 'nested_cart' vs 'cart' because we told it exactly what each label means."_

---

### Part 4: Compare Other VLM Models (3 min)

**Show model variety** (optional, pick 1-2):

1. **Discard** the previous annotations
2. Load **Grounding DINO Base** or **YOLOE**
3. Add prompts: `cart`, `nested cart`
4. Run annotation - show results
5. Compare: these models detect "cart" but can't distinguish nested vs single

**Optional enhancements to show:**
- **CLIP/SigLIP few-shot filtering:** Add example crops in the few-shot bar, enable "Use few-shot CLIP filter" to reject false positives visually
- **SAM2 refinement:** Enable "Refine boxes with SAM2" to tighten bounding boxes using segmentation masks
- **VQA verification:** Enable "Verify detections with VQA" to ask a second model yes/no questions about each detection
- **Multi-scale inference:** Enable for better detection at different object sizes

**Key point:** _"The tool supports multiple models. Pick the best one for your task: Qwen2.5-VL for understanding complex concepts, YOLOE for speed, Grounding DINO for simple object detection."_

---

### Part 5: Video Annotation Workflow (2 min)

1. Click **"Open Video"** in the workspace section
2. Select a video file
3. Configure extraction: "Every 30 frames" (1 fps from 30fps video)
4. Frames are extracted and loaded into the viewer
5. Switch to VLM tab, set scope to **"All"**
6. Run VLM annotation across all frames
7. Navigate through annotated frames with A/D

**Key point:** _"Full video support - extract frames, annotate (manually or with AI), then train."_

---

### Part 6: One-Click YOLO Training (3 min)

**Switch to the Train tab.**

1. Click **"Refresh Data Count"** - shows "X/Y annotated | Z classes"
2. Set training parameters:
   - Base model: **YOLOv8n (nano)** for quick demo
   - Epochs: **10** (for demo speed)
   - Batch size: **16**
   - Image size: **640**
   - Device auto-detected (GPU if available)
3. Click **"Start Training"**
4. Show live progress: epoch count, box_loss, mAP50
5. Training auto-splits data into train/val, generates YAML, runs training

**Key point:** _"From raw images to a trained custom YOLO model, all within the same tool. The trained best.pt can then be used back in the Auto Annotate dialog to annotate more data - creating a feedback loop."_

---

### Part 7: The Full Loop (1 min summary)

Show the complete workflow diagram:

```
Raw Images/Video
      |
      v
[Manual Annotation] or [VLM Auto-Annotation (Qwen2.5-VL/YOLOE/etc.)]
      |
      v
[Review & Correct] <-- Active Learning captures corrections
      |
      v
[Train YOLO Model] --> best.pt
      |
      v
[Use trained model for Auto-Annotate on new data]
      |
      v
  Repeat cycle --> Better model each iteration
```

---

## Key Talking Points

### Why This Tool?
1. **All-in-one:** Annotate, auto-label, and train without switching tools
2. **Model flexibility:** Multiple detection models, VQA verification models, embedding models, SAM2 variants
3. **Context-aware VLM detection:** Qwen2.5-VL understands natural-language label definitions — critical for domain-specific detection
4. **Video-native:** Direct video-to-annotation pipeline for both YOLO and VLM workflows
5. **Active learning:** System learns from your corrections automatically

### Model Comparison
| Capability | Qwen2.5-VL | Grounding DINO | YOLOE | OWL-ViT |
|-----------|------------|---------------|-------|---------|
| Understands label descriptions | Yes | No | No | No |
| Runs locally (no API) | Yes | Yes | Yes | Yes |
| Speed per image | ~1-2s | ~0.5s | ~0.2s | ~0.5s |
| Fine-grained classification | Good | Poor | Poor | Poor |
| VRAM needed | ~5GB | ~4GB | ~2GB | ~3GB |

### Technical Highlights
- YOLO format output (class_idx xc yc w h) - industry standard
- Multi-class mapping in one pass
- Live training dashboard with mAP50, losses, learning rate
- Resizable control panel for flexible workflows

---

## Backup / Contingency

- If a model takes too long to load: have YOLOE pre-loaded as backup, show that workflow instead
- If GPU not available: use Florence-2 on CPU (smaller model)
- If video extraction is slow: have pre-extracted frames ready in a folder
- If training takes too long: set epochs=5, show the progress bar updating, explain it would normally run longer
- Pre-annotate a few images before the demo so the Train tab shows data available immediately
