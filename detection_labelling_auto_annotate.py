"""
VLM-based Auto-Annotation module for detection_labelling.py

Supports:
  - Zero-shot / prompt-based bounding box prediction (Grounding DINO, OWL-ViT)
  - Few-shot learning via CLIP similarity filtering
  - Embeds directly into the annotation tool's left panel (no separate window)

Dependencies:
    pip install transformers torch torchvision pillow numpy
"""

import os
import json
import base64
import io
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image
import numpy as np


# ── Model Registry ─────────────────────────────────────────────────────────────

GROUNDING_DINO_MODELS = {
    "Grounding DINO Base": "IDEA-Research/grounding-dino-base",
    "Grounding DINO Tiny": "IDEA-Research/grounding-dino-tiny",
}

OWLVIT_MODELS = {
    "OWL-ViT Base": "google/owlvit-base-patch32",
    "OWL-ViT Large": "google/owlvit-large-patch14",
    "OWL-v2 Base": "google/owlv2-base-patch16-ensemble",
}

# YOLOE models use ultralytics — select a .pt file at load time
YOLOE_MODELS = {
    "YOLOE (select .pt file)": "yoloe",
}

FLORENCE2_MODELS = {
    "Florence-2 Base": "microsoft/Florence-2-base",
    "Florence-2 Large": "microsoft/Florence-2-large",
}

CLAUDE_MODELS = {
    "Claude": "claude",
}

QWEN_VL_MODELS = {
    "Qwen2.5-VL 3B (~5GB)": ("qwen2vl-detect", "Qwen/Qwen2.5-VL-3B-Instruct"),
}

ALL_MODELS = {**CLAUDE_MODELS, **QWEN_VL_MODELS, **YOLOE_MODELS, **FLORENCE2_MODELS, **GROUNDING_DINO_MODELS, **OWLVIT_MODELS}

SAM2_MODELS = {
    "SAM2 Tiny": "sam2_t.pt",
    "SAM2 Small": "sam2_s.pt",
    "SAM2 Base": "sam2_b.pt",
    "SAM2 Large": "sam2_l.pt",
}
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
ENCODER_CHOICES = {"CLIP": CLIP_MODEL_ID, "SigLIP": SIGLIP_MODEL_ID}


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Few-Shot Example Store ──────────────────────────────────────────────────────

class FewShotStore:
    """Stores example crops + CLIP embeddings per label. Persists to JSON."""

    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.examples: dict = {}
        self._clip_model = None
        self._clip_processor = None
        self._clip_device = "cpu"
        if save_path and os.path.exists(save_path):
            self._load(save_path)

    def _load(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.examples = data.get("examples", {})
        except Exception as e:
            print(f"FewShotStore: failed to load {path}: {e}")

    def save(self, path: str = None):
        p = path or self.save_path
        if p:
            try:
                with open(p, "w") as f:
                    json.dump({"version": 1, "examples": self.examples}, f)
            except Exception as e:
                print(f"FewShotStore: failed to save {p}: {e}")

    def load_encoder(self, encoder_name: str = "CLIP", device: str = None, progress_cb=None) -> bool:
        """Load CLIP or SigLIP as the image encoder for few-shot matching."""
        device = device or _get_device()
        model_id = ENCODER_CHOICES.get(encoder_name, CLIP_MODEL_ID)
        try:
            from transformers import AutoProcessor, AutoModel
            if progress_cb:
                progress_cb(f"Loading {encoder_name}...")
            self._clip_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self._clip_model = AutoModel.from_pretrained(
                model_id, low_cpu_mem_usage=False
            ).to(device)
            self._clip_model.eval()
            self._clip_device = device
            self._encoder_name = encoder_name
            # Recompute all embeddings with the new encoder
            self._recompute_all_embeddings()
            self.save()
            return True
        except Exception as e:
            print(f"{encoder_name} load failed: {e}")
            return False

    # Keep backward compat
    def load_clip(self, device: str = None, progress_cb=None) -> bool:
        return self.load_encoder("CLIP", device, progress_cb)

    def encoder_loaded(self) -> bool:
        return self._clip_model is not None

    def clip_loaded(self) -> bool:
        return self._clip_model is not None

    def _recompute_missing_embeddings(self):
        for label, entries in self.examples.items():
            for entry in entries:
                if "embedding" not in entry and "thumbnail_b64" in entry:
                    try:
                        img = self._b64_to_pil(entry["thumbnail_b64"])
                        emb = self._compute_clip_embedding(img)
                        if emb is not None:
                            entry["embedding"] = emb.tolist()
                    except Exception:
                        pass

    def _recompute_all_embeddings(self):
        """Recompute ALL embeddings (used when switching between CLIP and SigLIP).
        Removes old embeddings first to avoid dimension mismatches."""
        for label, entries in self.examples.items():
            for entry in entries:
                # Remove stale embedding from previous encoder
                entry.pop("embedding", None)
                if "thumbnail_b64" in entry:
                    try:
                        img = self._b64_to_pil(entry["thumbnail_b64"])
                        emb = self._compute_clip_embedding(img)
                        if emb is not None:
                            entry["embedding"] = emb.tolist()
                    except Exception:
                        pass

    def add_example(self, label: str, crop: Image.Image, image_path: str = "", bbox=None):
        if label not in self.examples:
            self.examples[label] = []
        thumb = crop.copy()
        thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
        entry = {
            "thumbnail_b64": self._pil_to_b64(thumb),
            "image_path": image_path,
            "bbox": list(bbox) if bbox else [],
        }
        emb = self._compute_clip_embedding(crop)
        if emb is not None:
            entry["embedding"] = emb.tolist()
        self.examples[label].append(entry)
        self.save()

    def remove_label(self, label: str):
        if label in self.examples:
            del self.examples[label]
            self.save()

    def get_labels(self) -> list:
        return list(self.examples.keys())

    def total_count(self) -> int:
        return sum(len(v) for v in self.examples.values())

    def summary(self) -> str:
        parts = [f"{lbl}: {len(exs)}" for lbl, exs in self.examples.items()]
        return ", ".join(parts) if parts else "none"

    def get_mean_embedding(self, label: str) -> np.ndarray:
        embeddings = [
            np.array(e["embedding"]) for e in self.examples.get(label, [])
            if "embedding" in e
        ]
        if not embeddings:
            return None
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-8 else mean

    def score_crop(self, label: str, crop: Image.Image) -> float:
        mean_emb = self.get_mean_embedding(label)
        if mean_emb is None:
            return 1.0
        crop_emb = self._compute_clip_embedding(crop)
        if crop_emb is None:
            return 1.0
        # Dimension mismatch = embeddings from different encoder (CLIP vs SigLIP)
        if mean_emb.shape[0] != crop_emb.shape[0]:
            return 1.0  # pass through, can't compare
        return float(np.dot(mean_emb, crop_emb))

    def score_crops_batch(self, label: str, crops: list) -> list:
        """Batch-score multiple crops against a label. Much faster than per-crop scoring."""
        if not crops:
            return []
        mean_emb = self.get_mean_embedding(label)
        if mean_emb is None or self._clip_model is None:
            return [1.0] * len(crops)
        import torch
        try:
            scores = []
            batch_size = 8
            for i in range(0, len(crops), batch_size):
                batch = crops[i:i + batch_size]
                inputs = self._clip_processor(images=batch, return_tensors="pt").to(self._clip_device)
                with torch.no_grad():
                    feats = self._clip_model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                embs = feats.cpu().numpy()
                for emb in embs:
                    if emb.shape[0] != mean_emb.shape[0]:
                        scores.append(1.0)
                    else:
                        scores.append(float(np.dot(mean_emb, emb)))
            return scores
        except Exception:
            return [1.0] * len(crops)

    def _compute_clip_embedding(self, img: Image.Image) -> np.ndarray:
        if self._clip_model is None:
            return None
        import torch
        try:
            inputs = self._clip_processor(images=img, return_tensors="pt").to(self._clip_device)
            with torch.no_grad():
                feat = self._clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()[0]
        except Exception as e:
            print(f"CLIP embedding failed: {e}")
            return None

    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _b64_to_pil(b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── VLM Annotator ──────────────────────────────────────────────────────────────

class VLMAnnotator:
    """Wraps YOLOE, Florence-2, Grounding DINO, and OWL-ViT for text-prompted detection."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._model_key = None
        self._model_type = None  # "yoloe", "florence2", "grounding-dino", or "owlvit"
        self._device = "cpu"
        self._yoloe_pt_path = None
        self._sam2_model = None
        self._sam2_variant = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Free GPU memory by unloading the current model."""
        import gc
        self._model = None
        self._processor = None
        self._model_key = None
        self._model_type = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    def load(self, model_key: str, device: str = None, progress_cb=None,
             yoloe_pt_path: str = None):
        device = device or _get_device()
        self._device = device
        self._model_key = model_key
        model_id = ALL_MODELS[model_key]

        # Handle tuple entries (type_tag, hf_model_id)
        if isinstance(model_id, tuple):
            type_tag, hf_id = model_id
        else:
            type_tag, hf_id = model_id, model_id

        if type_tag == "claude":
            self._load_claude(progress_cb)
        elif type_tag == "qwen2vl-detect":
            self._load_qwen_vl(model_key, hf_id, device, progress_cb)
        elif type_tag == "qwen3.5-detect":
            self._load_qwen35(model_key, hf_id, device, progress_cb)
        elif type_tag == "yoloe":
            self._load_yoloe(yoloe_pt_path, device, progress_cb)
        elif "florence" in hf_id.lower():
            self._load_florence2(model_key, hf_id, device, progress_cb)
        else:
            self._load_hf(model_key, hf_id, device, progress_cb)

    def _load_yoloe(self, pt_path, device, progress_cb):
        if not pt_path:
            raise ValueError("No .pt file selected for YOLOE.")
        try:
            from ultralytics import YOLOE
        except ImportError:
            from ultralytics import YOLO as YOLOE
        self._model_type = "yoloe"
        self._yoloe_pt_path = pt_path
        if progress_cb:
            progress_cb(f"Loading YOLOE from {os.path.basename(pt_path)}...")
        self._model = YOLOE(pt_path)
        if device == "cuda":
            self._model.to("cuda")
        self._processor = None  # not used for YOLOE

    def _load_hf(self, model_key, model_id, device, progress_cb):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._model_type = "grounding-dino" if "grounding-dino" in model_id.lower() else "owlvit"
        if progress_cb:
            progress_cb(f"Loading {model_key}...")
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        if progress_cb:
            progress_cb(f"Loading weights...")
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self._model.eval()

    def _load_florence2(self, model_key, model_id, device, progress_cb):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        self._model_type = "florence2"
        if progress_cb:
            progress_cb(f"Loading {model_key}...")
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        if progress_cb:
            progress_cb("Loading weights...")
        # Load with compatibility workarounds for Florence-2 custom code
        load_kwargs = dict(trust_remote_code=True)
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            self._patch_florence2_config(config)
            load_kwargs["config"] = config
        except Exception:
            pass
        # Use eager attention to avoid _supports_sdpa errors
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", **load_kwargs).to(device)
        except (TypeError, ValueError):
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs).to(device)
        self._model.eval()
        # Patch model config after load
        self._patch_florence2_config(self._model.config)
        if hasattr(self._model, "generation_config"):
            self._patch_florence2_config(self._model.generation_config)

    @staticmethod
    def _patch_florence2_config(config):
        """Patch Florence-2 config for newer transformers compatibility.
        Covers all nested sub-configs including Florence2LanguageConfig."""
        attrs = ("forced_bos_token_id", "forced_eos_token_id",
                 "suppress_tokens", "begin_suppress_tokens")
        configs_to_patch = [config]
        for sub_name in ("text_config", "language_config", "vision_config",
                         "projection_config", "decoder"):
            sub = getattr(config, sub_name, None)
            if sub is not None:
                configs_to_patch.append(sub)
        for cfg in configs_to_patch:
            for attr in attrs:
                if not hasattr(cfg, attr):
                    setattr(cfg, attr, None)

    def _load_qwen_vl(self, model_key, model_id, device, progress_cb):
        """Load Qwen2.5-VL for grounded object detection."""
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as Qwen2_5_VLForConditionalGeneration
        self._model_type = "qwen2vl-detect"
        if progress_cb:
            progress_cb(f"Loading {model_key} processor...")
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        if progress_cb:
            progress_cb("Loading weights (fp16)...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            self._model = self._model.float()
        self._model.eval()
        if progress_cb:
            progress_cb(f"Loaded: {model_key}")

    def _load_qwen35(self, model_key, model_id, device, progress_cb):
        """Load Qwen3.5-4B for image-based object detection."""
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as Qwen3_5ForConditionalGeneration
        self._model_type = "qwen3.5-detect"
        if progress_cb:
            progress_cb(f"Loading {model_key} processor...")
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        if progress_cb:
            progress_cb("Loading weights (fp16)...")
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        if progress_cb:
            progress_cb(f"Loaded: {model_key}")

    def _predict_qwen35(self, image, prompts, threshold, w, h,
                         claude_context: str = ""):
        """Use Qwen3.5-4B to detect objects via JSON output prompting."""
        import re
        import torch

        prompt_text = ", ".join(p.strip() for p in prompts if p.strip())
        labels_list = json.dumps([p.strip() for p in prompts if p.strip()])

        context_section = ""
        if claude_context and claude_context.strip():
            context_section = (
                f"\nLabel definitions:\n{claude_context.strip()}\n"
                f"Use these definitions strictly.\n"
            )

        text_prompt = (
            f"Detect all instances of: {prompt_text}.{context_section}\n"
            f"Labels MUST be one of: {labels_list}\n"
            f"Return ONLY a JSON array with each element having:\n"
            f'- "label": exact string from the list above\n'
            f'- "box": [x1, y1, x2, y2] as fractions of image size (0.0-1.0)\n'
            f'- "score": confidence 0.0-1.0\n'
            f"No explanation, no reasoning, just the JSON array. If nothing found return []."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an object detection assistant. Output only valid JSON arrays. No explanations."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt",
            padding=True
        )
        # Move only supported keys to model device
        inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=1024, do_sample=False)

        gen_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        result_text = self._processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        print(f"[Qwen3.5-4B] raw_output: {result_text[:400]}")

        # Strip thinking tags if present
        clean = result_text.strip()
        think_end = clean.find("</think>")
        if think_end >= 0:
            clean = clean[think_end + 8:].strip()

        # Parse JSON from response
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        try:
            detections_raw = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r'\[.*?\]', clean, re.DOTALL)
            if match:
                try:
                    detections_raw = json.loads(match.group())
                except json.JSONDecodeError:
                    print(f"Qwen3.5-4B parse error. Raw:\n{result_text[:500]}")
                    return []
            else:
                # Fall back to coordinate pattern parsing (same as Qwen2.5-VL)
                all_dets = []
                for label in prompts:
                    all_dets.extend(
                        self._parse_qwen_boxes(result_text, label.strip(), w, h, threshold))
                return all_dets

        if not isinstance(detections_raw, list):
            print(f"Qwen3.5-4B: expected list, got {type(detections_raw).__name__}")
            return []

        dets = []
        for obj in detections_raw:
            if not isinstance(obj, dict):
                continue
            box = obj.get("box", obj.get("bbox", []))
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                vals = [float(v) for v in box]
            except (ValueError, TypeError):
                continue
            if any(v > 1.5 for v in vals):
                vals = [v / 1000.0 for v in vals]
            x1_f, y1_f, x2_f, y2_f = vals
            score = float(obj.get("score", 0.8))
            if score < threshold:
                continue
            dets.append({
                "label": obj.get("label", "object"),
                "score": score,
                "bbox": (
                    max(0.0, x1_f * w), max(0.0, y1_f * h),
                    min(float(w), x2_f * w), min(float(h), y2_f * h),
                ),
            })
        return dets

    def _predict_qwen_vl(self, image, prompts, threshold, w, h,
                          claude_context: str = ""):
        """Use Qwen2.5-VL to detect objects — runs one prompt at a time for reliability."""
        import re
        import torch
        import io
        import base64 as b64mod
        try:
            from qwen_vl_utils import process_vision_info
            _has_qwen_utils = True
        except ImportError:
            _has_qwen_utils = False

        # Prepare image once as base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        img_b64 = b64mod.b64encode(buf.getvalue()).decode()

        all_dets = []

        # Run each prompt separately for better grounding accuracy
        for prompt_label in prompts:
            label = prompt_label.strip()
            if not label:
                continue

            # Build context for this label
            context_line = ""
            if claude_context and claude_context.strip():
                for line in claude_context.strip().splitlines():
                    low = line.lower().strip()
                    norm_label = label.lower().replace("_", " ").replace("-", " ")
                    norm_line = low.replace("_", " ").replace("-", " ")
                    if norm_label in norm_line or norm_line.startswith(norm_label):
                        context_line = f' Definition: {line.strip().split("=", 1)[-1].strip()}' if "=" in line else f" ({line.strip()})"
                        break

            text_prompt = (
                f"Detect and locate every instance of \"{label.replace('_', ' ')}\""
                f"{context_line} in this image. "
                f"For each one, output the bounding box coordinates."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            if _has_qwen_utils:
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt"
                ).to(self._model.device)
            else:
                inputs = self._processor(
                    text=[text], images=[image], padding=True, return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=512, do_sample=False)

            gen_ids_trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            result_text = self._processor.batch_decode(
                gen_ids_trimmed, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)[0]

            print(f"[Qwen2.5-VL] prompt='{label}' raw_output: {result_text[:300]}")

            # Parse bounding boxes from output
            dets = self._parse_qwen_boxes(result_text, label, w, h, threshold)
            all_dets.extend(dets)

        return all_dets

    @staticmethod
    def _parse_qwen_boxes(text, label, w, h, threshold):
        """Parse bounding boxes from Qwen2.5-VL output.
        Handles multiple formats:
          1. Native: <|box_start|>(x1,y1),(x2,y2)<|box_end|>  (0-1000 scale)
          2. Tuple pairs: (x1,y1),(x2,y2) or (x1, y1), (x2, y2)
          3. Bracket coords: [x1, y1, x2, y2]
          4. JSON array with box key
        """
        import re

        dets = []

        # Pattern 1: Qwen native box tokens (0-1000 scale)
        native_pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
        for m in re.finditer(native_pattern, text):
            x1, y1, x2, y2 = [int(v) / 1000.0 for v in m.groups()]
            dets.append({
                "label": label, "score": 0.85,
                "bbox": (max(0, x1 * w), max(0, y1 * h),
                         min(float(w), x2 * w), min(float(h), y2 * h)),
            })

        if dets:
            return dets

        # Pattern 2: Coordinate tuple pairs  (x1, y1), (x2, y2)
        pair_pattern = r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,?\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)'
        for m in re.finditer(pair_pattern, text):
            vals = [float(v) for v in m.groups()]
            # Determine scale: >1.5 means 0-1000 Qwen format
            if any(v > 1.5 for v in vals):
                vals = [v / 1000.0 for v in vals]
            x1, y1, x2, y2 = vals
            if x2 > x1 and y2 > y1:
                dets.append({
                    "label": label, "score": 0.80,
                    "bbox": (max(0, x1 * w), max(0, y1 * h),
                             min(float(w), x2 * w), min(float(h), y2 * h)),
                })

        if dets:
            return dets

        # Pattern 3: Bracket coordinates [x1, y1, x2, y2]
        bracket_pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
        for m in re.finditer(bracket_pattern, text):
            vals = [float(v) for v in m.groups()]
            if any(v > 1.5 for v in vals):
                vals = [v / 1000.0 for v in vals]
            x1, y1, x2, y2 = vals
            if x2 > x1 and y2 > y1:
                dets.append({
                    "label": label, "score": 0.80,
                    "bbox": (max(0, x1 * w), max(0, y1 * h),
                             min(float(w), x2 * w), min(float(h), y2 * h)),
                })

        if dets:
            return dets

        # Pattern 4: JSON fallback
        try:
            clean = text.strip()
            if "```" in clean:
                clean = clean.split("```")[1] if clean.count("```") >= 2 else clean
                clean = clean.lstrip("json").strip()
            json_match = re.search(r'\[.*?\]', clean, re.DOTALL)
            if json_match:
                arr = json.loads(json_match.group())
                for obj in arr:
                    box = obj.get("box", obj.get("bbox", []))
                    if len(box) != 4:
                        continue
                    vals = [float(v) for v in box]
                    if any(v > 1.5 for v in vals):
                        vals = [v / 1000.0 for v in vals]
                    x1, y1, x2, y2 = vals
                    if x2 > x1 and y2 > y1:
                        dets.append({
                            "label": obj.get("label", label),
                            "score": float(obj.get("score", 0.80)),
                            "bbox": (max(0, x1 * w), max(0, y1 * h),
                                     min(float(w), x2 * w), min(float(h), y2 * h)),
                        })
        except Exception:
            pass

        return dets

    def _load_claude(self, progress_cb):
        """Claude API needs no local model — just verify the SDK is importable."""
        self._model_type = "claude"
        self._processor = None
        if progress_cb:
            progress_cb("Checking Claude SDK...")
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "claude_agent_sdk not installed.\n"
                "Install it with: pip install claude-agent-sdk"
            )
        self._model = "claude"  # sentinel so is_loaded() returns True
        if progress_cb:
            progress_cb("Claude ready")

    def _predict_claude(self, image, prompts, threshold, w, h,
                         claude_context: str = ""):
        """Use Claude via claude_agent_sdk to detect objects in the image."""
        import asyncio
        import tempfile

        from claude_agent_sdk import query, ClaudeAgentOptions

        # Save image to a temp file so Claude can read it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp, format="JPEG", quality=95)
            tmp_path = tmp.name.replace("\\", "/")

        prompt_text = ", ".join(p.strip() for p in prompts if p.strip())
        labels_list = json.dumps([p.strip() for p in prompts if p.strip()])

        # Build context section describing what each label means
        context_section = ""
        if claude_context and claude_context.strip():
            context_section = (
                f"\nIMPORTANT — Label definitions (read carefully before annotating):\n"
                f"{claude_context.strip()}\n\n"
                f"Use these definitions strictly. Only label something as a class if it "
                f"clearly matches the definition above. If unsure, use a lower confidence "
                f"score. Do NOT label an object as a class if it doesn't match the definition.\n"
            )

        prompt_msg = (
            f"Read the image at {tmp_path} and detect objects.\n"
            f"You must classify each detected object as one of: {labels_list}\n"
            f"{context_section}"
            f"Return ONLY a JSON array where each element has:\n"
            f'- "label": MUST be one of exactly these strings: {labels_list} — '
            f"use the EXACT string, do not rephrase or add spaces\n"
            f'- "box": [x1, y1, x2, y2] as fractions of image width/height (0.0-1.0)\n'
            f'- "score": confidence score between 0.0 and 1.0 '
            f"(use lower scores like 0.3-0.6 when uncertain)\n\n"
            f'Example: [{{"label":"{prompts[0].strip()}","box":[0.1,0.2,0.4,0.7],"score":0.95}}]\n'
            f"No markdown, no explanation, just the JSON array. "
            f"If no objects found, return an empty array []. "
            f"Be precise — only annotate objects you are confident about."
        )

        async def _run():
            result_text = ""
            async for message in query(
                prompt=prompt_msg,
                options=ClaudeAgentOptions(allowed_tools=["Read"]),
            ):
                if hasattr(message, "result") and message.result:
                    result_text = message.result
                    break
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text = block.text
                            break
                    if result_text:
                        break
            return result_text

        # Run async query in a new event loop (we're in a worker thread)
        try:
            loop = asyncio.new_event_loop()
            result_text = loop.run_until_complete(_run())
        finally:
            loop.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Parse JSON response
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        try:
            detections_raw = json.loads(clean)
        except json.JSONDecodeError:
            print(f"Claude response parse error. Raw:\n{result_text[:500]}")
            return []

        dets = []
        for obj in detections_raw:
            box = obj.get("box", [])
            if len(box) != 4:
                continue
            x1_f, y1_f, x2_f, y2_f = [float(v) for v in box]
            score = float(obj.get("score", 0.9))
            if score < threshold:
                continue
            dets.append({
                "label": obj.get("label", "object"),
                "score": score,
                "bbox": (
                    max(0.0, x1_f * w), max(0.0, y1_f * h),
                    min(float(w), x2_f * w), min(float(h), y2_f * h),
                ),
            })
        return dets

    def predict(self, image: Image.Image, prompts: list, threshold: float = 0.3,
                nms_iou: float = 0.5, claude_context: str = "") -> list:
        if not self._model:
            raise RuntimeError("Model not loaded.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        if self._model_type == "claude":
            dets = self._predict_claude(image, prompts, threshold, w, h,
                                         claude_context=claude_context)
        elif self._model_type == "qwen2vl-detect":
            dets = self._predict_qwen_vl(image, prompts, threshold, w, h,
                                          claude_context=claude_context)
        elif self._model_type == "qwen3.5-detect":
            dets = self._predict_qwen35(image, prompts, threshold, w, h,
                                         claude_context=claude_context)
        elif self._model_type == "yoloe":
            dets = self._predict_yoloe(image, prompts, threshold, w, h)
        elif self._model_type == "florence2":
            dets = self._predict_florence2(image, prompts, w, h)
        elif self._model_type == "grounding-dino":
            dets = self._predict_gdino(image, prompts, threshold, w, h)
        else:
            dets = self._predict_owlvit(image, prompts, threshold, w, h)
        return self._apply_nms(dets, nms_iou)

    def _predict_yoloe(self, image, prompts, threshold, w, h):
        import numpy as np
        # Set text prompts as classes
        names = list(prompts)
        try:
            # New API: set_classes takes only names
            self._model.set_classes(names)
        except TypeError:
            # Old API: set_classes needs text_pe from get_text_pe
            try:
                text_pe = self._model.get_text_pe(names)
                self._model.set_classes(names, text_pe)
            except Exception:
                # Fallback: just set names directly
                self._model.set_classes(names)

        # Save PIL image to a temp array for ultralytics
        img_array = np.array(image)

        results = self._model.predict(img_array, conf=threshold, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            boxes_xyxy = r.boxes.xyxy.cpu().tolist()
            scores = r.boxes.conf.cpu().tolist()
            cls_ids = r.boxes.cls.cpu().tolist()
            for box, score, cls_id in zip(boxes_xyxy, scores, cls_ids):
                x1, y1, x2, y2 = box
                idx = int(cls_id)
                label = names[idx] if idx < len(names) else "object"
                dets.append({
                    "label": label,
                    "score": float(score),
                    "bbox": (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                })
        return dets

    def _predict_florence2(self, image, prompts, w, h):
        import torch
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        text = ". ".join(p.strip().rstrip(".") for p in prompts if p.strip()) + "."
        prompt = task + text

        # Process text and image separately then merge — most reliable across versions
        text_inputs = self._processor.tokenizer(
            prompt, return_tensors="pt", padding=True).to(self._device)
        try:
            img_inputs = self._processor.image_processor(
                image, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self._device)
        except Exception:
            # Fallback: use the full processor
            combined = self._processor(text=prompt, images=image, return_tensors="pt").to(self._device)
            pixel_values = combined.get("pixel_values")
            if pixel_values is None:
                raise RuntimeError("Florence-2 processor failed to produce pixel_values")

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=text_inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=1024,
            )
        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        try:
            parsed = self._processor.post_process_generation(
                decoded, task=task, image_size=(w, h))
        except TypeError:
            parsed = self._processor.post_process_generation(
                decoded, task=task, image_size=image.size)

        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])

        dets = []
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            # Normalize: ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue  # skip degenerate boxes
            dets.append({
                "label": label.strip(),
                "score": 1.0,
                "bbox": (x1, y1, x2, y2),
            })
        return dets

    def _predict_gdino(self, image, prompts, threshold, w, h):
        import torch
        text = ". ".join(p.strip().rstrip(".") for p in prompts if p.strip()) + "."
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # post_process_grounded_object_detection moved between versions
        post_proc = None
        for candidate in [self._processor, getattr(self._processor, "image_processor", None), self._model]:
            if candidate and hasattr(candidate, "post_process_grounded_object_detection"):
                post_proc = candidate.post_process_grounded_object_detection
                break

        if post_proc is None:
            print("Grounding DINO: post_process_grounded_object_detection not found")
            return []

        try:
            results = post_proc(
                outputs, inputs.input_ids,
                threshold=threshold,
                target_sizes=[(h, w)],
            )
        except TypeError:
            try:
                results = post_proc(
                    outputs, inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=max(0.1, threshold * 0.8),
                    target_sizes=[(h, w)],
                )
            except TypeError:
                results = post_proc(
                    outputs, inputs.input_ids,
                    target_sizes=[(h, w)],
                )
        dets = []
        if results:
            r = results[0]
            for box, score, label in zip(r.get("boxes", []), r.get("scores", []), r.get("labels", [])):
                x1, y1, x2, y2 = [float(v) for v in box]
                dets.append({
                    "label": label if isinstance(label, str) else str(label),
                    "score": float(score),
                    "bbox": (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                })
        return dets

    def _predict_owlvit(self, image, prompts, threshold, w, h):
        import torch
        text_queries = [[f"a photo of {p}" for p in prompts]]
        inputs = self._processor(text=text_queries, images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([[h, w]], device=self._device)

        # post_process_object_detection moved between versions:
        # Old: processor.post_process_object_detection(...)
        # New: processor.image_processor.post_process_object_detection(...)
        # Also: model.post_process_object_detection(...) in some versions
        post_proc = None
        for candidate in [self._processor, getattr(self._processor, "image_processor", None), self._model]:
            if candidate and hasattr(candidate, "post_process_object_detection"):
                post_proc = candidate.post_process_object_detection
                break

        if post_proc is None:
            print("OWL-ViT: post_process_object_detection not found on processor, image_processor, or model")
            return []

        try:
            results = post_proc(
                outputs=outputs, threshold=threshold, target_sizes=target_sizes)
        except TypeError:
            try:
                results = post_proc(
                    outputs, target_sizes=target_sizes, score_threshold=threshold)
            except TypeError:
                results = post_proc(
                    outputs=outputs, target_sizes=target_sizes)

        dets = []
        if results:
            r = results[0]
            for box, score, label_idx in zip(r.get("boxes", []), r.get("scores", []), r.get("labels", [])):
                x1, y1, x2, y2 = [float(v) for v in box]
                idx = int(label_idx)
                label = prompts[idx] if idx < len(prompts) else "object"
                dets.append({
                    "label": label, "score": float(score),
                    "bbox": (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                })
        return dets

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.0

    def _apply_nms(self, dets, iou_threshold):
        if not dets:
            return []
        # Fast path: use torchvision for large detection sets
        if len(dets) >= 20:
            try:
                import torch
                from torchvision.ops import nms as torch_nms
                boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
                scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
                keep_idx = torch_nms(boxes, scores, iou_threshold).cpu().tolist()
                return [dets[i] for i in keep_idx]
            except Exception:
                pass
        # Fallback: pure Python O(n²)
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        keep, suppressed = [], set()
        for i, d in enumerate(dets):
            if i in suppressed:
                continue
            keep.append(d)
            for j in range(i + 1, len(dets)):
                if j not in suppressed and self._iou(d["bbox"], dets[j]["bbox"]) > iou_threshold:
                    suppressed.add(j)
        return keep

    # ── SAM2 Refinement ───────────────────────────────────────────────────────

    def load_sam2(self, variant_key: str, device: str = None, progress_cb=None):
        if self._sam2_model and self._sam2_variant == variant_key:
            return  # already loaded
        from ultralytics import SAM
        device = device or _get_device()
        pt_name = SAM2_MODELS[variant_key]
        if progress_cb:
            progress_cb(f"Loading {variant_key}...")
        self._sam2_model = SAM(pt_name)
        if device == "cuda":
            self._sam2_model.to("cuda")
        self._sam2_variant = variant_key

    def refine_with_sam2(self, image: Image.Image, detections: list,
                          tight_bbox: bool = False) -> list:
        """Run SAM2 on each detection bbox. Adds 'polygon' and optionally 'tight_bbox' keys."""
        if not self._sam2_model or not detections:
            return detections

        img_array = np.array(image)
        w, h = image.size
        bboxes = [list(d["bbox"]) for d in detections]

        try:
            results = self._sam2_model(img_array, bboxes=bboxes)
        except Exception as e:
            print(f"SAM2 inference failed: {e}")
            return detections

        for i, det in enumerate(detections):
            try:
                if results and results[0].masks is not None and i < len(results[0].masks.data):
                    mask = results[0].masks.data[i].cpu().numpy()
                    poly = self._mask_to_polygon(mask, w, h)
                    det["polygon"] = poly
                    if tight_bbox and poly and len(poly) >= 6:
                        xs = [poly[j] for j in range(0, len(poly), 2)]
                        ys = [poly[j] for j in range(1, len(poly), 2)]
                        det["bbox"] = (min(xs) * w, min(ys) * h,
                                       max(xs) * w, max(ys) * h)
            except Exception:
                pass
        return detections

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int) -> list:
        """Convert binary mask to normalized polygon [x1,y1,x2,y2,...] for YOLO seg format."""
        try:
            import cv2
        except ImportError:
            return None
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.005 * peri, True)
        if len(approx) < 3:
            return None
        poly = []
        for pt in approx:
            poly.append(float(pt[0][0]) / img_w)
            poly.append(float(pt[0][1]) / img_h)
        return poly

    # ── Multi-Scale + WBF ─────────────────────────────────────────────────────

    def predict_multiscale(self, image: Image.Image, prompts: list,
                            threshold: float, scales: list,
                            nms_iou: float = 0.5, wbf_iou: float = 0.55) -> list:
        """Run detection at multiple scales, fuse results with WBF."""
        w, h = image.size
        all_dets = []

        for scale in scales:
            if scale is None or scale >= max(w, h):
                scaled_img = image
                sx, sy = 1.0, 1.0
            else:
                ratio = scale / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                scaled_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
                sx, sy = w / new_w, h / new_h

            dets = self.predict(scaled_img, prompts, threshold, nms_iou)
            # Map boxes back to original image coordinates
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            all_dets.append(dets)

        return _apply_wbf(all_dets, w, h, iou_thr=wbf_iou)


def _apply_wbf(all_dets_per_scale: list, img_w: int, img_h: int,
                iou_thr: float = 0.55, skip_box_thr: float = 0.01) -> list:
    """Fuse detections from multiple passes using Weighted Boxes Fusion."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("ensemble-boxes not installed. pip install ensemble-boxes. Falling back to first scale.")
        return all_dets_per_scale[0] if all_dets_per_scale else []

    # Build label mapping
    all_labels = set()
    for dets in all_dets_per_scale:
        for d in dets:
            all_labels.add(d["label"])
    label_list = sorted(all_labels)
    label_to_idx = {l: i for i, l in enumerate(label_list)}

    boxes_list, scores_list, labels_list = [], [], []
    for dets in all_dets_per_scale:
        boxes, scores, labels = [], [], []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
            scores.append(d["score"])
            labels.append(label_to_idx[d["label"]])
        boxes_list.append(boxes if boxes else np.zeros((0, 4)))
        scores_list.append(scores if scores else [])
        labels_list.append(labels if labels else [])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    idx_to_label = {i: l for l, i in label_to_idx.items()}
    result = []
    for box, score, label_idx in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box
        result.append({
            "label": idx_to_label.get(int(label_idx), "object"),
            "score": float(score),
            "bbox": (x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h),
        })
    return result


# ── VLM Verifier (VQA-based spatial reasoning) ─────────────────────────────────

VERIFIER_MODELS = {
    "Florence-2 Base (~1GB)": ("florence2", "microsoft/Florence-2-base"),
    "Florence-2 Large (~1.5GB)": ("florence2", "microsoft/Florence-2-large"),
    "Qwen2-VL 2B (~5GB)": ("qwen2vl", "Qwen/Qwen2-VL-2B-Instruct"),
    "Qwen2.5-VL 3B (~5GB)": ("qwen2.5vl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "LLaVA-1.5 7B (Q4 GGUF ~4GB)": ("llava-gguf", "llava-gguf"),
    "moondream2 (~3.5GB)": ("moondream", "vikhyatk/moondream2"),
    "PaliGemma 3B (~6GB)": ("paligemma", "google/paligemma-3b-mix-224"),
}


class VLMVerifier:
    """Uses a VQA model to verify each detection crop with a yes/no question.
    Supports Florence-2, Qwen2-VL, moondream2, and PaliGemma."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = "cpu"
        self._model_type = None  # "florence2", "qwen2vl", "moondream", "paligemma"
        self._loaded_key = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Free GPU memory by unloading the VQA model."""
        import gc
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_type = None
        self._loaded_key = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    def load(self, model_key: str, device: str = None, progress_cb=None):
        import torch
        device = device or _get_device()
        self._device = device
        model_type, model_id = VERIFIER_MODELS[model_key]
        self._model_type = model_type
        self._loaded_key = model_key
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._dtype = dtype

        if progress_cb:
            progress_cb(f"Loading {model_key}...")

        if model_type == "florence2":
            self._load_florence2(model_id, device, dtype)
        elif model_type == "qwen2vl":
            self._load_qwen2vl(model_id, device, dtype)
        elif model_type == "qwen2.5vl":
            self._load_qwen25vl(model_id, device, dtype)
        elif model_type == "llava-gguf":
            self._load_llava_gguf(device)
        elif model_type == "moondream":
            self._load_moondream(model_id, device, dtype)
        elif model_type == "paligemma":
            self._load_paligemma(model_id, device, dtype)

    def _load_florence2(self, model_id, device, dtype):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        load_kwargs = dict(trust_remote_code=True, torch_dtype=dtype)
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            VLMAnnotator._patch_florence2_config(config)
            load_kwargs["config"] = config
        except Exception:
            pass
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", **load_kwargs).to(device)
        except (TypeError, ValueError):
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, **load_kwargs).to(device)
        self._model.eval()
        VLMAnnotator._patch_florence2_config(self._model.config)
        if hasattr(self._model, "generation_config"):
            VLMAnnotator._patch_florence2_config(self._model.generation_config)

    def _load_qwen2vl(self, model_id, device, dtype):
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as Qwen2VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype).to(device)
        self._model.eval()

    def _load_qwen25vl(self, model_id, device, dtype):
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as Qwen2_5_VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self._model_type = "qwen2.5vl"

    def _load_llava_gguf(self, device):
        """Load LLaVA-1.5 7B via llama-cpp-python with GGUF weights.
        Requires user to have set _llava_model_path and _llava_clip_path
        on this instance before calling load()."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for LLaVA GGUF.\n\n"
                "Install with:\n"
                "  pip install llama-cpp-python\n\n"
                "For GPU support:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )

        model_path = getattr(self, "_llava_model_path", None)
        clip_path = getattr(self, "_llava_clip_path", None)
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("LLaVA GGUF model file not selected.")
        if not clip_path or not os.path.exists(clip_path):
            raise RuntimeError("LLaVA mmproj clip file not selected.")

        n_gpu = -1 if device != "cpu" else 0
        chat_handler = Llava15ChatHandler(clip_model_path=clip_path, verbose=False)
        self._model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=n_gpu,
            logits_all=True,
            verbose=False,
        )
        self._model_type = "llava-gguf"
        self._processor = None

    def _verify_llava_gguf(self, crop, question):
        """Run LLaVA GGUF inference on a crop image."""
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        result = self._model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": question},
                ],
            }],
            max_tokens=50,
        )
        return result["choices"][0]["message"]["content"]

    def _load_moondream(self, model_id, device, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype).to(device)
        self._model.eval()

    def _load_paligemma(self, model_id, device, dtype):
        from transformers import AutoProcessor
        try:
            from transformers import PaliGemmaForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM as PaliGemmaForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype).to(device)
        self._model.eval()

    def load_from_existing(self, model, processor, device, model_type="florence2",
                           loaded_key=None):
        """Reuse an already-loaded model (avoids duplicate VRAM usage)."""
        self._model = model
        self._processor = processor
        self._device = device
        self._model_type = model_type
        self._loaded_key = loaded_key or f"{model_type} (reused)"

    def verify(self, crop: Image.Image, question: str) -> bool:
        """Ask a yes/no question about the crop. Returns True if answer contains 'yes'."""
        import torch
        crop = crop.convert("RGB")

        if self._model_type == "florence2":
            answer = self._verify_florence2(crop, question)
        elif self._model_type in ("qwen2vl", "qwen2.5vl"):
            answer = self._verify_qwen2vl(crop, question)
        elif self._model_type == "llava-gguf":
            answer = self._verify_llava_gguf(crop, question)
        elif self._model_type == "moondream":
            answer = self._verify_moondream(crop, question)
        elif self._model_type == "paligemma":
            answer = self._verify_paligemma(crop, question)
        else:
            return True
        return "yes" in answer.strip().lower()

    def ask(self, crop: Image.Image, question: str) -> str:
        """Ask a question about the crop. Returns the raw answer string."""
        crop = crop.convert("RGB")
        if self._model_type == "florence2":
            return self._verify_florence2(crop, question)
        elif self._model_type in ("qwen2vl", "qwen2.5vl"):
            return self._verify_qwen2vl(crop, question)
        elif self._model_type == "llava-gguf":
            return self._verify_llava_gguf(crop, question)
        elif self._model_type == "moondream":
            return self._verify_moondream(crop, question)
        elif self._model_type == "paligemma":
            return self._verify_paligemma(crop, question)
        return ""

    def _cast_inputs(self, inputs):
        """Cast pixel_values to match model dtype (fixes float32/float16 mismatch)."""
        import torch
        dtype = getattr(self, "_dtype", torch.float32)
        for key in ("pixel_values", "pixel_values_videos", "image_embeds"):
            val = inputs.get(key, None) if isinstance(inputs, dict) else getattr(inputs, key, None)
            if val is not None and hasattr(val, "to"):
                inputs[key] = val.to(dtype=dtype)
        return inputs

    def _verify_florence2(self, crop, question):
        import torch
        task = "<VQA>"
        prompt = task + question
        # Process text and image separately — most reliable across versions
        text_inputs = self._processor.tokenizer(
            prompt, return_tensors="pt", padding=True).to(self._device)
        try:
            img_inputs = self._processor.image_processor(crop, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self._device)
        except Exception:
            combined = self._processor(text=prompt, images=crop, return_tensors="pt").to(self._device)
            pixel_values = combined.get("pixel_values")
            if pixel_values is None:
                return ""
        pixel_values = self._cast_pixel_values(pixel_values)
        with torch.no_grad():
            ids = self._model.generate(
                input_ids=text_inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=20)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]

    def _cast_pixel_values(self, pixel_values):
        """Cast pixel_values to match model dtype."""
        import torch
        dtype = getattr(self, "_dtype", torch.float32)
        if pixel_values is not None and hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(dtype=dtype)
        return pixel_values

    def _verify_qwen2vl(self, crop, question):
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": question}
        ]}]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[crop], return_tensors="pt",
                                  padding=True).to(self._device)
        inputs = self._cast_inputs(inputs)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=20)
        out_ids = ids[:, input_len:]
        return self._processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    def _verify_moondream(self, crop, question):
        import torch
        # moondream2 API varies between versions
        if hasattr(self._model, "encode_image"):
            enc_img = self._model.encode_image(crop)
            with torch.no_grad():
                answer = self._model.answer_question(enc_img, question, self._tokenizer)
        elif hasattr(self._model, "query"):
            answer = self._model.query(crop, question)["answer"]
        else:
            # Fallback: try as a standard generate pipeline
            inputs = self._tokenizer(question, return_tensors="pt").to(self._device)
            with torch.no_grad():
                ids = self._model.generate(**inputs, max_new_tokens=20)
            answer = self._tokenizer.decode(ids[0], skip_special_tokens=True)
        return answer

    def _verify_paligemma(self, crop, question):
        import torch
        inputs = self._processor(text=question, images=crop, return_tensors="pt").to(self._device)
        inputs = self._cast_inputs(inputs)
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=20)
        return self._processor.decode(ids[0], skip_special_tokens=True)


# ── VLM Controller (embedded in left panel, no separate window) ────────────────

class VLMController:
    """
    Manages VLM state and builds UI that embeds directly into the annotation tool's
    left panel. Two pieces of UI:
      1. VLM tab content (model, prompts, run) — inside the notebook
      2. Few-shot bar (label + add button) — persistent, always visible
    """

    def __init__(self, tool):
        self.tool = tool
        self.root = tool.root
        self.annotator = VLMAnnotator()
        self.few_shot = FewShotStore(save_path=self._resolve_store_path())
        self._running = False
        self._prompts = []  # list of (prompt_text, class_idx_str)
        self._vlm_predictions = {}  # {image_path: [{"class_idx", "bbox"}]} for active learning
        self.verifier = VLMVerifier()

    def _resolve_store_path(self) -> str:
        labels_dir = getattr(self.tool, "labels_dir", "") or ""
        if labels_dir and os.path.isdir(labels_dir):
            return os.path.join(labels_dir, ".vlm_few_shot_examples.json")
        return None

    # ── Build VLM tab (goes inside the notebook) ──────────────────────────────

    def build_vlm_tab(self, notebook: ttk.Notebook):
        """Creates the 'VLM' tab inside the left panel's notebook."""
        BG = "#f5f5f5"
        FONT = ("Segoe UI", 9)
        FONT_BOLD = ("Segoe UI", 9, "bold")
        FONT_SM = ("Segoe UI", 8)

        outer = tk.Frame(notebook, bg=BG)
        notebook.add(outer, text="  VLM  ")

        # Scrollable canvas wrapper
        _canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        _vsb = ttk.Scrollbar(outer, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tab = tk.Frame(_canvas, bg=BG)
        _canvas_win = _canvas.create_window((0, 0), window=tab, anchor="nw")

        def _on_frame_configure(event):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        tab.bind("<Configure>", _on_frame_configure)

        def _on_canvas_configure(event):
            _canvas.itemconfig(_canvas_win, width=event.width)
        _canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        _canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # -- Model section --
        tk.Label(tab, text="Model", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8, pady=(8, 2))

        model_row = tk.Frame(tab, bg=BG)
        model_row.pack(fill=tk.X, padx=8, pady=2)
        self.model_var = tk.StringVar(value=list(ALL_MODELS.keys())[0])
        ttk.Combobox(model_row, textvariable=self.model_var,
                      values=list(ALL_MODELS.keys()), state="readonly", width=24).pack(side=tk.LEFT)

        load_row = tk.Frame(tab, bg=BG)
        load_row.pack(fill=tk.X, padx=8, pady=2)
        self.device_var = tk.StringVar(value=_get_device())
        ttk.Combobox(load_row, textvariable=self.device_var,
                      values=["cpu", "cuda", "mps"], state="readonly", width=6).pack(side=tk.LEFT)
        self.btn_load = tk.Button(load_row, text="Load Model", bg="#b8d4f0",
                                   font=FONT, relief=tk.GROOVE, command=self._load_model_thread)
        self.btn_load.pack(side=tk.LEFT, padx=4)
        self.btn_unload = tk.Button(load_row, text="Unload", bg="#f0d5d5",
                                     font=FONT_SM, relief=tk.GROOVE, command=self._unload_model)
        self.btn_unload.pack(side=tk.LEFT, padx=2)
        self.lbl_model_status = tk.Label(load_row, text="Not loaded", fg="gray", bg=BG, font=FONT_SM)
        self.lbl_model_status.pack(side=tk.LEFT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Prompts section --
        tk.Label(tab, text="Detection Prompts", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)

        prompt_entry_row = tk.Frame(tab, bg=BG)
        prompt_entry_row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(prompt_entry_row, text="Text:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.prompt_entry = tk.Entry(prompt_entry_row, width=18, font=FONT)
        self.prompt_entry.pack(side=tk.LEFT, padx=4)
        self.prompt_entry.bind("<Return>", lambda _: self._add_prompt())

        cls_row = tk.Frame(tab, bg=BG)
        cls_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(cls_row, text="Map to:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.prompt_class_var = tk.StringVar()
        self.prompt_class_cb = ttk.Combobox(cls_row, textvariable=self.prompt_class_var,
                                             state="readonly", width=12)
        self.prompt_class_cb.pack(side=tk.LEFT, padx=4)
        tk.Button(cls_row, text="Add", command=self._add_prompt, width=5,
                  font=FONT, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.LEFT, padx=2)
        self._refresh_class_dropdown()

        self.prompt_listbox = tk.Listbox(tab, height=4, font=FONT, exportselection=False,
                                          relief=tk.GROOVE, bd=1)
        self.prompt_listbox.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(tab, text="Remove Selected", command=self._remove_prompt,
                  font=FONT, relief=tk.GROOVE, bg="#f0d5d5").pack(padx=8, anchor=tk.W, pady=2)

        # -- Claude Context (visible only for Claude model) --
        self.claude_context_frame = tk.Frame(tab, bg=BG)
        self.claude_context_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(self.claude_context_frame, text="Label context (Claude / Qwen2.5-VL):",
                 bg=BG, font=FONT_SM, fg="#555").pack(anchor=tk.W)
        self.claude_context_entry = tk.Text(self.claude_context_frame, width=30, height=4,
                                             font=FONT_SM, relief=tk.GROOVE, bd=1, wrap=tk.WORD)
        self.claude_context_entry.pack(fill=tk.X, pady=2)
        self.claude_context_entry.insert("1.0",
            "nested_cart = a shopping cart with one or more other carts stacked/nested inside it\n"
            "cart = a single standalone shopping cart with NO other cart inside")

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Settings --
        tk.Label(tab, text="Settings", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)

        thresh_row = tk.Frame(tab, bg=BG)
        thresh_row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(thresh_row, text="Confidence:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.30)
        self.lbl_thresh = tk.Label(thresh_row, text="0.30", bg=BG, font=FONT, width=4)
        self.lbl_thresh.pack(side=tk.RIGHT)
        ttk.Scale(thresh_row, from_=0.05, to=0.95, variable=self.threshold_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_thresh.config(text=f"{float(v):.2f}")).pack(side=tk.RIGHT, padx=4)

        # Scope
        tk.Label(tab, text="Scope:", bg=BG, font=FONT).pack(anchor=tk.W, padx=8, pady=(4, 0))
        scope_frame = tk.Frame(tab, bg=BG)
        scope_frame.pack(fill=tk.X, padx=8, pady=2)
        self.scope_var = tk.StringVar(value="current")
        for val, txt in [("current", "Current"), ("all", "All"), ("unannotated", "New only"), ("custom", "Custom dir")]:
            ttk.Radiobutton(scope_frame, text=txt, variable=self.scope_var, value=val).pack(side=tk.LEFT, padx=1)

        self.merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Keep existing annotations",
                         variable=self.merge_var).pack(anchor=tk.W, padx=8, pady=2)

        self.use_fewshot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Use few-shot CLIP filter",
                         variable=self.use_fewshot_var).pack(anchor=tk.W, padx=8, pady=2)

        sim_row = tk.Frame(tab, bg=BG)
        sim_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(sim_row, text="CLIP similarity:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.sim_threshold_var = tk.DoubleVar(value=0.70)
        self.lbl_sim = tk.Label(sim_row, text="0.70", bg=BG, font=FONT, width=4)
        self.lbl_sim.pack(side=tk.RIGHT)
        ttk.Scale(sim_row, from_=0.0, to=1.0, variable=self.sim_threshold_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_sim.config(text=f"{float(v):.2f}")).pack(side=tk.RIGHT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- SAM2 Refinement --
        tk.Label(tab, text="SAM2 Refinement", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)
        self.use_sam2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Refine boxes with SAM2",
                         variable=self.use_sam2_var).pack(anchor=tk.W, padx=8, pady=1)

        sam2_row = tk.Frame(tab, bg=BG)
        sam2_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(sam2_row, text="SAM2 model:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.sam2_model_var = tk.StringVar(value="SAM2 Base")
        ttk.Combobox(sam2_row, textvariable=self.sam2_model_var,
                      values=list(SAM2_MODELS.keys()), state="readonly", width=12).pack(side=tk.LEFT, padx=4)

        self.sam2_tight_bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Tight bbox from mask",
                         variable=self.sam2_tight_bbox_var).pack(anchor=tk.W, padx=8, pady=1)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Multi-Scale + WBF --
        tk.Label(tab, text="Multi-Scale + Fusion", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)
        self.use_multiscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Multi-scale inference + WBF",
                         variable=self.use_multiscale_var).pack(anchor=tk.W, padx=8, pady=1)

        scales_row = tk.Frame(tab, bg=BG)
        scales_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(scales_row, text="Scales:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.scales_entry = tk.Entry(scales_row, width=16, font=FONT)
        self.scales_entry.insert(0, "640,1024,original")
        self.scales_entry.pack(side=tk.LEFT, padx=4)

        wbf_row = tk.Frame(tab, bg=BG)
        wbf_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(wbf_row, text="WBF IoU:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.wbf_iou_var = tk.DoubleVar(value=0.55)
        self.lbl_wbf = tk.Label(wbf_row, text="0.55", bg=BG, font=FONT, width=4)
        self.lbl_wbf.pack(side=tk.RIGHT)
        ttk.Scale(wbf_row, from_=0.1, to=0.95, variable=self.wbf_iou_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_wbf.config(text=f"{float(v):.2f}")).pack(side=tk.RIGHT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- VQA Verification --
        tk.Label(tab, text="VQA Verification", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)
        self.use_verify_var = tk.BooleanVar(value=False)
        self.use_verify_var.trace_add("write", self._on_verify_toggle)
        ttk.Checkbutton(tab, text="Verify detections with VQA",
                         variable=self.use_verify_var).pack(anchor=tk.W, padx=8, pady=1)

        vmodel_row = tk.Frame(tab, bg=BG)
        vmodel_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(vmodel_row, text="VQA model:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.verify_model_var = tk.StringVar(value=list(VERIFIER_MODELS.keys())[0])
        ttk.Combobox(vmodel_row, textvariable=self.verify_model_var,
                      values=list(VERIFIER_MODELS.keys()), state="readonly",
                      width=22).pack(side=tk.LEFT, padx=4)

        vqa_row = tk.Frame(tab, bg=BG)
        vqa_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(vqa_row, text="Question:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.vqa_question_entry = tk.Entry(vqa_row, width=26, font=FONT)
        self.vqa_question_entry.insert(0, "Does this image show {prompt}? Answer yes or no.")
        self.vqa_question_entry.pack(side=tk.LEFT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Few-Shot bar (inside VLM tab) --
        fewshot_bar = self.build_fewshot_bar(tab)
        fewshot_bar.pack(fill=tk.X, padx=4, pady=2)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Run --
        run_row = tk.Frame(tab, bg=BG)
        run_row.pack(fill=tk.X, padx=8, pady=4)
        self.btn_run = tk.Button(run_row, text="Run VLM Annotation", bg="#a8d5a2",
                                  font=("Segoe UI", 10, "bold"), relief=tk.GROOVE,
                                  command=self._run_annotation_thread)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_stop = tk.Button(run_row, text="Stop", bg="#f0d5d5",
                                   font=("Segoe UI", 10), relief=tk.GROOVE,
                                   state=tk.DISABLED, command=self._stop_annotation)
        self.btn_stop.pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(tab, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=8, pady=2)
        self.lbl_status = tk.Label(tab, text="Ready", bg=BG, fg="#555",
                                    font=FONT, anchor=tk.W, wraplength=240)
        self.lbl_status.pack(fill=tk.X, padx=8)

        return outer

    # ── Build few-shot bar (embedded inside VLM tab) ─────────────────────────

    def build_fewshot_bar(self, parent: tk.Frame):
        """Creates the few-shot bar inside the VLM tab."""
        BG_BAR = "#e3eaf2"
        FONT = ("Segoe UI", 9)
        FONT_BOLD = ("Segoe UI", 9, "bold")

        bar = tk.Frame(parent, bg=BG_BAR, bd=1, relief=tk.GROOVE)

        tk.Label(bar, text="Few-Shot Examples", bg=BG_BAR,
                 font=FONT_BOLD).pack(anchor=tk.W, padx=8, pady=(6, 2))

        # Label entry + Add button
        add_row = tk.Frame(bar, bg=BG_BAR)
        add_row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(add_row, text="Label:", bg=BG_BAR, font=FONT).pack(side=tk.LEFT)
        self.example_label_entry = tk.Entry(add_row, width=10, font=FONT)
        self.example_label_entry.pack(side=tk.LEFT, padx=4)
        tk.Button(add_row, text="+ Add BBox", bg="#f5edb8", font=FONT,
                  relief=tk.GROOVE, command=self._add_current_bbox_as_example).pack(side=tk.LEFT, padx=2)

        # Info + actions row
        info_row = tk.Frame(bar, bg=BG_BAR)
        info_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.lbl_examples_info = tk.Label(info_row, text="0 examples", bg=BG_BAR,
                                           font=FONT, fg="#555")
        self.lbl_examples_info.pack(side=tk.LEFT)
        tk.Button(info_row, text="Clear All", font=FONT, relief=tk.GROOVE,
                  bg="#f0d5d5", command=self._clear_all_examples).pack(side=tk.RIGHT)

        # Encoder selection (CLIP or SigLIP)
        encoder_row = tk.Frame(bar, bg=BG_BAR)
        encoder_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(encoder_row, text="Encoder:", bg=BG_BAR, font=FONT).pack(side=tk.LEFT)
        self.encoder_var = tk.StringVar(value="SigLIP")
        ttk.Combobox(encoder_row, textvariable=self.encoder_var,
                      values=list(ENCODER_CHOICES.keys()), state="readonly",
                      width=8).pack(side=tk.LEFT, padx=4)
        self.btn_clip = tk.Button(encoder_row, text="Load", font=FONT, relief=tk.GROOVE,
                                   bg="#b8d4f0", command=self._load_encoder_thread)
        self.btn_clip.pack(side=tk.LEFT, padx=2)
        self.lbl_clip = tk.Label(encoder_row, text="", bg=BG_BAR, font=FONT, fg="gray")
        self.lbl_clip.pack(side=tk.RIGHT, padx=2)

        # Save / Load examples
        persist_row = tk.Frame(bar, bg=BG_BAR)
        persist_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Button(persist_row, text="Save Examples", font=FONT, relief=tk.GROOVE,
                  bg="#dce6f0", command=self._save_examples_as).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(persist_row, text="Load Examples", font=FONT, relief=tk.GROOVE,
                  bg="#dce6f0", command=self._load_examples_from).pack(side=tk.LEFT)
        self.lbl_save_path = tk.Label(persist_row, text="", bg=BG_BAR,
                                       font=("Segoe UI", 7), fg="#888")
        self.lbl_save_path.pack(side=tk.LEFT, padx=4)

        # Active learning toggle
        al_row = tk.Frame(bar, bg=BG_BAR)
        al_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.active_learning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(al_row, text="Active learning (auto-capture corrections)",
                         variable=self.active_learning_var).pack(side=tk.LEFT)

        self._update_examples_info()
        # Auto-load if save path exists
        if self.few_shot.save_path and os.path.exists(self.few_shot.save_path):
            self.lbl_save_path.config(text=os.path.basename(self.few_shot.save_path))
        return bar

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_model_thread(self):
        model_key = self.model_var.get()
        model_id = ALL_MODELS.get(model_key, "")

        # YOLOE: ask user to pick a .pt file first (must be on main thread)
        yoloe_pt_path = None
        if model_id == "yoloe":
            yoloe_pt_path = filedialog.askopenfilename(
                title="Select YOLOE .pt Model File",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
            if not yoloe_pt_path:
                return  # user cancelled

        self.btn_load.config(state=tk.DISABLED)
        self.lbl_model_status.config(text="Loading...", fg="orange")
        threading.Thread(target=self._load_model,
                         args=(model_key, yoloe_pt_path), daemon=True).start()

    def _load_model(self, model_key, yoloe_pt_path=None):
        try:
            self.annotator.load(
                model_key, device=self.device_var.get(),
                progress_cb=lambda msg: self.root.after(0, lambda m=msg: self.lbl_model_status.config(text=m, fg="orange")),
                yoloe_pt_path=yoloe_pt_path,
            )
            name = os.path.basename(yoloe_pt_path) if yoloe_pt_path else model_key
            self.root.after(0, lambda: self.lbl_model_status.config(
                text=f"Loaded: {name}", fg="green"))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: self.lbl_model_status.config(text="Error", fg="red"))
            self.root.after(0, lambda m=err_msg: messagebox.showerror("Model Error", m))
        finally:
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def _unload_model(self):
        """Unload the current detection model to free GPU memory."""
        self.annotator.unload()
        self.lbl_model_status.config(text="Not loaded (VRAM freed)", fg="gray")

    def _load_encoder_thread(self):
        self.btn_clip.config(state=tk.DISABLED)
        encoder = self.encoder_var.get()
        self.lbl_clip.config(text=f"Loading {encoder}...", fg="orange")
        threading.Thread(target=self._load_encoder, args=(encoder,), daemon=True).start()

    def _load_encoder(self, encoder_name):
        ok = self.few_shot.load_encoder(encoder_name, device=self.device_var.get())
        if ok:
            self.root.after(0, lambda: self.lbl_clip.config(text=f"{encoder_name} OK", fg="green"))
        else:
            self.root.after(0, lambda: self.lbl_clip.config(text="Failed", fg="red"))
        self.root.after(0, lambda: self.btn_clip.config(state=tk.NORMAL))

    # ── Prompt Management ──────────────────────────────────────────────────────

    def _refresh_class_dropdown(self):
        classes = getattr(self.tool, "classes", [])
        values = [f"{i}: {c}" for i, c in enumerate(classes)] if classes else ["No classes"]
        self.prompt_class_cb["values"] = values
        if values:
            self.prompt_class_cb.current(0)

    def _add_prompt(self):
        prompt = self.prompt_entry.get().strip()
        cls_str = self.prompt_class_var.get()
        if not prompt:
            return
        if not cls_str or "No classes" in cls_str:
            messagebox.showwarning("No Class", "Load classes in the Annotate tab first.")
            return
        display = f"{prompt}  →  {cls_str}"
        self._prompts.append((prompt, cls_str))
        self.prompt_listbox.insert(tk.END, display)
        self.prompt_entry.delete(0, tk.END)

    def _remove_prompt(self):
        sel = self.prompt_listbox.curselection()
        if sel:
            idx = sel[0]
            self.prompt_listbox.delete(idx)
            if idx < len(self._prompts):
                self._prompts.pop(idx)

    def _get_prompt_configs(self) -> list:
        """Returns list of (prompt_text, class_idx, class_name)."""
        configs = []
        for prompt, cls_str in self._prompts:
            try:
                cls_idx = int(cls_str.split(":")[0])
                cls_name = ":".join(cls_str.split(":")[1:]).strip()
                configs.append((prompt, cls_idx, cls_name))
            except (ValueError, IndexError):
                continue
        return configs

    # ── Few-Shot Example Management ────────────────────────────────────────────

    def _add_current_bbox_as_example(self):
        label = self.example_label_entry.get().strip()
        if not label:
            messagebox.showwarning("No Label", "Enter a label name first.")
            return
        if not getattr(self.tool, "current_image", None):
            messagebox.showwarning("No Image", "No image loaded.")
            return
        if getattr(self.tool, "selected_rect_idx", None) is None:
            messagebox.showwarning("No Selection", "Draw and select a bounding box first.")
            return

        ann = self.tool.annotations[self.tool.selected_rect_idx]
        x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            messagebox.showwarning("Too Small", "Selected box is too small.")
            return

        crop = self.tool.current_image.crop((x1, y1, x2, y2))
        img_path = ""
        if self.tool.image_files and self.tool.images_dir:
            img_path = os.path.join(self.tool.images_dir, self.tool.image_files[self.tool.current_index])

        # Update store path if labels dir changed
        new_path = self._resolve_store_path()
        if new_path and new_path != self.few_shot.save_path:
            self.few_shot.save_path = new_path

        self.few_shot.add_example(label, crop, image_path=img_path, bbox=[x1, y1, x2, y2])
        self._update_examples_info()

        lbl_count = len(self.few_shot.examples.get(label, []))
        self.lbl_status.config(text=f"Added example for '{label}' ({lbl_count} total for this label)")

    def _clear_all_examples(self):
        if not self.few_shot.examples:
            return
        if messagebox.askyesno("Clear", "Remove ALL few-shot examples?"):
            self.few_shot.examples.clear()
            self.few_shot.save()
            self._update_examples_info()

    def _update_examples_info(self):
        total = self.few_shot.total_count()
        if total == 0:
            self.lbl_examples_info.config(text="0 examples")
        else:
            self.lbl_examples_info.config(text=f"{total} ex ({self.few_shot.summary()})")

    def _save_examples_as(self):
        """Save few-shot examples (thumbnails + CLIP embeddings) to a JSON file."""
        if self.few_shot.total_count() == 0:
            messagebox.showinfo("Nothing to Save", "No few-shot examples to save.")
            return
        # Default to labels_dir if set
        initial_dir = getattr(self.tool, "labels_dir", "") or ""
        path = filedialog.asksaveasfilename(
            title="Save Few-Shot Examples",
            initialdir=initial_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        self.few_shot.save_path = path
        self.few_shot.save()
        self.lbl_save_path.config(text=os.path.basename(path))
        messagebox.showinfo("Saved",
            f"Saved {self.few_shot.total_count()} examples to:\n{path}\n\n"
            "This file contains thumbnails and CLIP embeddings (if computed).\n"
            "You can reload it in a future session with 'Load Examples'.")

    def _load_examples_from(self):
        """Load few-shot examples from a previously saved JSON file."""
        initial_dir = getattr(self.tool, "labels_dir", "") or ""
        path = filedialog.askopenfilename(
            title="Load Few-Shot Examples",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        old_count = self.few_shot.total_count()
        self.few_shot._load(path)
        self.few_shot.save_path = path
        new_count = self.few_shot.total_count()
        self._update_examples_info()
        self.lbl_save_path.config(text=os.path.basename(path))
        messagebox.showinfo("Loaded",
            f"Loaded {new_count} examples from:\n{os.path.basename(path)}\n\n"
            f"Labels: {', '.join(self.few_shot.get_labels()) or 'none'}")

    # ── Annotation Runner ──────────────────────────────────────────────────────

    def _run_annotation_thread(self):
        if not self.annotator.is_loaded():
            messagebox.showwarning("Model Not Loaded", "Load a VLM model first.")
            return
        prompt_configs = self._get_prompt_configs()
        if not prompt_configs:
            messagebox.showwarning("No Prompts", "Add at least one prompt.")
            return
        use_fewshot = self.use_fewshot_var.get()
        if use_fewshot:
            if not self.few_shot.clip_loaded():
                messagebox.showwarning("CLIP Not Loaded", "Load CLIP first (in the few-shot bar).")
                return
            if not self.few_shot.get_labels():
                messagebox.showwarning("No Examples", "Add few-shot examples first.")
                return

        img_paths = self._collect_target_images()
        if img_paths is None:
            return
        if not img_paths:
            messagebox.showinfo("No Images", "No images found.")
            return

        # Read all feature flags from UI
        use_sam2 = self.use_sam2_var.get()
        sam2_variant = self.sam2_model_var.get()
        tight_bbox = self.sam2_tight_bbox_var.get()
        use_multiscale = self.use_multiscale_var.get()
        wbf_iou = self.wbf_iou_var.get()
        active_learning = self.active_learning_var.get()
        use_verify = self.use_verify_var.get()
        verify_model_key = self.verify_model_var.get()
        vqa_question = self.vqa_question_entry.get().strip()

        # LLaVA GGUF: prompt for model files on main thread before background work
        if use_verify and "GGUF" in verify_model_key:
            needs_files = (not self.verifier.is_loaded()
                           or self.verifier._loaded_key != verify_model_key)
            if needs_files:
                model_path = filedialog.askopenfilename(
                    title="Select LLaVA GGUF Model File (e.g. llava-v1.5-7b-Q4_K_M.gguf)",
                    filetypes=[("GGUF Model", "*.gguf"), ("All Files", "*.*")])
                if not model_path:
                    return
                clip_path = filedialog.askopenfilename(
                    title="Select mmproj Clip File (e.g. mmproj-model-f16.gguf)",
                    filetypes=[("GGUF Model", "*.gguf"), ("All Files", "*.*")])
                if not clip_path:
                    return
                self.verifier._llava_model_path = model_path
                self.verifier._llava_clip_path = clip_path

        # Read Claude context (only used when model is Claude)
        claude_context = self.claude_context_entry.get("1.0", tk.END).strip()

        # Parse scales
        scales = [None]  # default: original only
        if use_multiscale:
            try:
                raw = self.scales_entry.get().strip()
                scales = []
                for s in raw.split(","):
                    s = s.strip().lower()
                    scales.append(None if s in ("original", "orig", "none") else int(s))
            except ValueError:
                scales = [640, 1024, None]

        self._running = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_bar.configure(maximum=len(img_paths), value=0)

        threading.Thread(
            target=self._run_annotation,
            args=(img_paths, prompt_configs, use_fewshot,
                  use_sam2, sam2_variant, tight_bbox,
                  use_multiscale, scales, wbf_iou, active_learning,
                  use_verify, verify_model_key, vqa_question,
                  claude_context),
            daemon=True
        ).start()

    @staticmethod
    def _read_existing_boxes(img_path, w, h):
        """Read existing YOLO annotation boxes as (x1, y1, x2, y2) pixel coords."""
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            return []
        boxes = []
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        boxes.append((x1, y1, x2, y2))
                    except ValueError:
                        continue
        except OSError:
            pass
        return boxes

    def _stop_annotation(self):
        """Stop the running VLM annotation after the current image."""
        self._running = False
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Stopping after current image...")

    def _on_verify_toggle(self, *_args):
        """Unload VQA model when verification is unchecked to free VRAM."""
        if not self.use_verify_var.get() and self.verifier.is_loaded():
            # Don't unload if the verifier is sharing the detector's model
            if self.verifier._model is self.annotator._model:
                self.verifier._model = None
                self.verifier._processor = None
                self.verifier._loaded_key = None
                self.verifier._model_type = None
            else:
                self.verifier.unload()
                self.lbl_status.config(text="VQA model unloaded (VRAM freed)")

    def _collect_target_images(self) -> list:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        scope = self.scope_var.get()

        if scope == "current":
            if not getattr(self.tool, "image_files", None) or not self.tool.images_dir:
                messagebox.showwarning("No Image", "No image loaded.")
                return None
            return [os.path.join(self.tool.images_dir, self.tool.image_files[self.tool.current_index])]

        elif scope in ("all", "unannotated"):
            if not getattr(self.tool, "images_dir", None):
                messagebox.showwarning("No Dir", "No images directory loaded.")
                return None
            paths = []
            for root_d, _, files in os.walk(self.tool.images_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        full = os.path.join(root_d, f)
                        if scope == "unannotated":
                            if os.path.exists(os.path.splitext(full)[0] + ".txt"):
                                continue
                        paths.append(full)
            return sorted(paths)

        elif scope == "custom":
            d = filedialog.askdirectory(title="Select Directory to Annotate")
            if not d:
                return None
            paths = []
            for root_d, _, files in os.walk(d):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        paths.append(os.path.join(root_d, f))
            return sorted(paths)

        return None

    def _run_annotation(self, img_paths, prompt_configs, use_fewshot,
                         use_sam2, sam2_variant, tight_bbox,
                         use_multiscale, scales, wbf_iou, active_learning,
                         use_verify=False, verify_model_key="", vqa_question="",
                         claude_context=""):
        threshold = self.threshold_var.get()
        sim_threshold = self.sim_threshold_var.get() if use_fewshot else 0.0
        merge = self.merge_var.get()
        is_current = self.scope_var.get() == "current"
        prompts = [p for p, _, _ in prompt_configs]

        # Lazy-load SAM2 if needed
        if use_sam2:
            try:
                self.root.after(0, lambda: self.lbl_status.config(text="Loading SAM2..."))
                self.annotator.load_sam2(sam2_variant, progress_cb=lambda m:
                    self.root.after(0, lambda msg=m: self.lbl_status.config(text=msg)))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda m=err_msg: messagebox.showerror("SAM2 Error", m))
                self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
                self._running = False
                return

        # Lazy-load verifier if needed (or reload if model changed)
        if use_verify:
            needs_load = (not self.verifier.is_loaded()
                          or self.verifier._loaded_key != verify_model_key)
            if needs_load:
                try:
                    self.root.after(0, lambda: self.lbl_status.config(text="Loading verifier..."))
                    # Reuse detector model if same family is already loaded
                    vtype = VERIFIER_MODELS.get(verify_model_key, ("", ""))[0]
                    det_type = self.annotator._model_type
                    reused = False

                    # Florence-2 reuse
                    if (vtype == "florence2" and det_type == "florence2"
                            and self.annotator._model is not None):
                        self.verifier.load_from_existing(
                            self.annotator._model, self.annotator._processor,
                            self.annotator._device, model_type="florence2",
                            loaded_key=verify_model_key)
                        reused = True

                    # Qwen2.5-VL reuse: detector is qwen2vl-detect, verifier is qwen2.5vl
                    if (vtype == "qwen2.5vl" and det_type == "qwen2vl-detect"
                            and self.annotator._model is not None):
                        self.verifier.load_from_existing(
                            self.annotator._model, self.annotator._processor,
                            self.annotator._device, model_type="qwen2.5vl",
                            loaded_key=verify_model_key)
                        reused = True

                    if not reused:
                        self.verifier.load(verify_model_key, device=self.device_var.get())
                except Exception as e:
                    err_msg = str(e)
                    self.root.after(0, lambda m=err_msg: messagebox.showerror("Verifier Error", m))
                    self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
                    self._running = False
                    return

        total = len(img_paths)
        total_boxes = 0
        annotated = 0
        errors = 0
        current_img_anns = []

        for i, img_path in enumerate(img_paths):
            if not self._running:
                break
            try:
                self.root.after(0, lambda p=img_path:
                    self.lbl_status.config(text=f"Processing: {os.path.basename(p)}"))

                image = Image.open(img_path).convert("RGB")
                w, h = image.size

                # Step 1: Detection (optionally multi-scale)
                if use_multiscale and len(scales) > 1:
                    detections = self.annotator.predict_multiscale(
                        image, prompts, threshold, scales, wbf_iou=wbf_iou)
                else:
                    detections = self.annotator.predict(
                        image, prompts, threshold=threshold,
                        claude_context=claude_context)

                # Step 1.5: Deduplicate against existing annotations (IoU > 0.6)
                if merge and detections:
                    existing_boxes = self._read_existing_boxes(img_path, w, h)
                    if existing_boxes:
                        deduped = []
                        for det in detections:
                            dup = False
                            for eb in existing_boxes:
                                if self.annotator._iou(det["bbox"], eb) > 0.6:
                                    dup = True
                                    break
                            if not dup:
                                deduped.append(det)
                        detections = deduped

                # Step 2: Filter by class + pre-crop all detections
                matched_dets = []
                for det in detections:
                    cls_idx = self._match_detection_to_class(det["label"], prompt_configs)
                    if cls_idx is None:
                        continue
                    det["class_idx"] = cls_idx
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    # Ensure valid crop coordinates
                    cx1, cy1 = max(0, min(x1, x2)), max(0, min(y1, y2))
                    cx2, cy2 = min(w, max(x1, x2)), min(h, max(y1, y2))
                    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                        continue
                    det["_crop"] = image.crop((cx1, cy1, cx2, cy2))
                    matched_dets.append(det)

                # CLIP few-shot batch filtering
                if use_fewshot and sim_threshold > 0 and matched_dets:
                    labels = self.few_shot.get_labels()
                    crops = [d["_crop"] for d in matched_dets]
                    # Batch score all crops against each label, take max per crop
                    best_scores = [0.0] * len(crops)
                    for lbl in labels:
                        lbl_scores = self.few_shot.score_crops_batch(lbl, crops)
                        for idx, s in enumerate(lbl_scores):
                            if s > best_scores[idx]:
                                best_scores[idx] = s
                    new_anns = [d for d, s in zip(matched_dets, best_scores)
                                if s >= sim_threshold]
                else:
                    new_anns = matched_dets

                # Step 3: VQA verification (reuse pre-cropped images)
                if use_verify and vqa_question and new_anns:
                    verified = []
                    for ann in new_anns:
                        crop = ann["_crop"]
                        prompt_text = ann.get("label", "this object")
                        question = vqa_question.replace("{prompt}", prompt_text)
                        try:
                            if self.verifier.verify(crop, question):
                                verified.append(ann)
                        except Exception:
                            verified.append(ann)
                    new_anns = verified

                # Clean up temp crop references
                for ann in new_anns:
                    ann.pop("_crop", None)

                # Step 4: SAM2 refinement (adds 'polygon' key, optionally tightens bbox)
                if use_sam2 and new_anns:
                    self.root.after(0, lambda p=img_path:
                        self.lbl_status.config(text=f"SAM2 refining: {os.path.basename(p)}"))
                    new_anns = self.annotator.refine_with_sam2(
                        image, new_anns, tight_bbox=tight_bbox)

                # Step 4: Write label files
                txt_paths = set()
                txt_paths.add(os.path.splitext(img_path)[0] + ".txt")
                labels_dir = getattr(self.tool, "labels_dir", "") or ""
                if labels_dir:
                    img_basename = os.path.basename(img_path)
                    label_in_labels_dir = os.path.join(
                        labels_dir, os.path.splitext(img_basename)[0] + ".txt")
                    txt_paths.add(label_in_labels_dir)

                # 4a: Bounding box labels (YOLO detection format)
                bbox_lines = []
                for ann in new_anns:
                    x1, y1, x2, y2 = ann["bbox"]
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    xc, yc = (x1 / w) + bw / 2, (y1 / h) + bh / 2
                    bbox_lines.append(f"{ann['class_idx']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                for txt_path in txt_paths:
                    existing = []
                    if merge and os.path.exists(txt_path):
                        with open(txt_path, "r") as f:
                            existing = f.readlines()
                    if existing or bbox_lines:
                        with open(txt_path, "w") as f:
                            f.writelines(existing)
                            f.writelines(bbox_lines)

                # 4b: Segmentation labels (YOLO seg format) if SAM2 produced polygons
                if use_sam2:
                    seg_lines = []
                    for ann in new_anns:
                        poly = ann.get("polygon")
                        if poly and len(poly) >= 6:
                            coords = " ".join(f"{p:.6f}" for p in poly)
                            seg_lines.append(f"{ann['class_idx']} {coords}\n")
                    if seg_lines:
                        for txt_path in txt_paths:
                            seg_path = txt_path.replace(".txt", "_seg.txt")
                            seg_existing = []
                            if merge and os.path.exists(seg_path):
                                with open(seg_path, "r") as f:
                                    seg_existing = f.readlines()
                            with open(seg_path, "w") as f:
                                f.writelines(seg_existing)
                                f.writelines(seg_lines)

                # Step 5: Active learning — store predictions
                if active_learning and new_anns:
                    self._vlm_predictions[img_path] = [
                        {"class_idx": a["class_idx"], "bbox": tuple(a["bbox"])}
                        for a in new_anns
                    ]

                total_boxes += len(new_anns)
                if new_anns:
                    annotated += 1
                    if is_current:
                        current_img_anns = [{"class_idx": a["class_idx"], "bbox": a["bbox"]}
                                            for a in new_anns]

            except Exception as e:
                print(f"Error on {img_path}: {e}")
                errors += 1

            self.root.after(0, lambda v=i+1: self.progress_bar.configure(value=v))

        def on_done():
            self._running = False
            self.btn_run.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            msg = f"Done: {annotated}/{total} images, {total_boxes} boxes"
            if use_sam2:
                msg += " + seg masks"
            if errors:
                msg += f" ({errors} errors)"
            self.lbl_status.config(text=msg)

            if is_current and current_img_anns:
                if merge:
                    self.tool.annotations.extend(current_img_anns)
                else:
                    self.tool.annotations = current_img_anns
                self.tool.redraw_annotations()
                self.tool.save_annotations()
            # Always reload current image to show annotations (works for all scopes)
            if getattr(self.tool, "image_files", None):
                self.tool.load_current_image()

            messagebox.showinfo("Complete", msg)

        self.root.after(0, on_done)

    @staticmethod
    def _match_detection_to_class(detected_label: str, prompt_configs: list):
        detected_label = detected_label.lower().strip()
        norm_det = detected_label.replace("_", " ").replace("-", " ")
        for prompt, cls_idx, cls_name in prompt_configs:
            pl = prompt.lower().strip()
            norm_pl = pl.replace("_", " ").replace("-", " ")
            if (detected_label == pl or norm_det == norm_pl
                    or detected_label in pl or pl in detected_label
                    or norm_det in norm_pl or norm_pl in norm_det):
                return cls_idx
        return prompt_configs[0][1] if prompt_configs else None

    # ── Active Learning ────────────────────────────────────────────────────────

    def check_corrections_on_navigate(self):
        """Called when user navigates away from an image. Compares current annotations
        with VLM predictions and auto-adds corrected crops as few-shot examples."""
        if not self.active_learning_var.get():
            return
        if not getattr(self.tool, "current_image", None):
            return
        if not self.tool.image_files or not self.tool.images_dir:
            return

        img_path = os.path.join(
            self.tool.images_dir, self.tool.image_files[self.tool.current_index])
        stored = self._vlm_predictions.get(img_path, [])
        if not stored:
            return

        current = self.tool.annotations
        if not current:
            return

        added = 0
        for curr_ann in current:
            # Find if this annotation is a correction of a VLM prediction
            best_iou = 0.0
            best_stored = None
            for s in stored:
                iou = VLMAnnotator._iou(curr_ann["bbox"], s["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_stored = s

            # If IoU is 0.3-0.9, it's a correction (user moved/resized the box)
            # If IoU > 0.9, it's unchanged. If < 0.3, it's a new user annotation.
            if best_stored and 0.3 < best_iou < 0.9:
                x1, y1, x2, y2 = [int(v) for v in curr_ann["bbox"]]
                w, h = self.tool.current_image.size
                crop = self.tool.current_image.crop(
                    (max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
                cls_idx = curr_ann["class_idx"]
                cls_name = (self.tool.classes[cls_idx]
                            if cls_idx < len(self.tool.classes) else f"class_{cls_idx}")

                new_path = self._resolve_store_path()
                if new_path and new_path != self.few_shot.save_path:
                    self.few_shot.save_path = new_path

                self.few_shot.add_example(cls_name, crop, image_path=img_path,
                                           bbox=[x1, y1, x2, y2])
                added += 1

        if added > 0:
            self._update_examples_info()
            # Remove stored predictions for this image (processed)
            del self._vlm_predictions[img_path]


# ── Entry Point ─────────────────────────────────────────────────────────────────

def create_vlm_controller(tool) -> VLMController:
    """Create a VLMController attached to the given YOLOAnnotationTool."""
    return VLMController(tool)
