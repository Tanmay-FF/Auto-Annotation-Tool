"""
Smart Detect-Then-Classify Annotation Pipeline

Stage 1: Detect ALL instances of a base object (e.g. "cart") using VLM detector
Stage 2: Classify each detection with VQA (e.g. "single cart" vs "nested carts")
Stage 3 (optional): OWLv2 image-conditioned detection using example crops

Usage:
    from smart_annotator import create_smart_ui
    smart_ui = create_smart_ui(tool, vlm_controller)
    smart_ui.build_smart_tab(notebook)
"""

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

from detection_labelling_auto_annotate import (
    VLMAnnotator, VLMVerifier, FewShotStore,
    VERIFIER_MODELS, ALL_MODELS, SAM2_MODELS, _get_device,
)

LETTERS = "ABCDEFGHIJKLMNOP"


@dataclass
class ClassRule:
    class_idx: int
    class_name: str
    description: str
    letter: str = ""


# ── Smart Annotator (no UI dependency) ─────────────────────────────────────────

class SmartAnnotator:
    """Two-stage detect-then-classify pipeline."""

    def __init__(self):
        self._detector = None
        self._classifier = None
        self._owlv2_model = None
        self._owlv2_processor = None
        self._owlv2_device = "cpu"

    def set_detector(self, annotator: VLMAnnotator):
        self._detector = annotator

    def set_classifier(self, verifier: VLMVerifier):
        self._classifier = verifier

    def detect_and_classify(self, image: Image.Image, base_prompt: str,
                             class_rules: list, threshold: float = 0.3,
                             context_pad: float = 0.3,
                             nms_iou: float = 0.5) -> list:
        """
        Stage 1: Detect all instances of base_prompt.
        Stage 2: Classify each with VQA multiple-choice.
        Returns: [{"class_idx", "class_name", "bbox", "score", "vqa_answer"}]
        """
        if not self._detector or not self._detector.is_loaded():
            raise RuntimeError("Detection model not loaded.")
        if not self._classifier or not self._classifier.is_loaded():
            raise RuntimeError("VQA classifier model not loaded.")

        image = image.convert("RGB")
        w, h = image.size

        # Stage 1: Broad detection
        detections = self._detector.predict(image, [base_prompt], threshold, nms_iou)
        if not detections:
            return []

        # Build classification prompt
        prompt = self._build_classification_prompt(class_rules)

        # Stage 2: Classify each detection
        results = []
        for det in detections:
            padded = self._pad_bbox(det["bbox"], (w, h), context_pad)
            crop = image.crop(padded)
            answer = self._classifier.ask(crop, prompt)
            cls_idx, letter = self._parse_classification_answer(answer, class_rules)
            if cls_idx >= 0:
                results.append({
                    "class_idx": cls_idx,
                    "class_name": next((r.class_name for r in class_rules
                                        if r.class_idx == cls_idx), ""),
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "vqa_answer": answer.strip(),
                    "vqa_letter": letter,
                })
        return results

    def detect_with_image_queries(self, image: Image.Image,
                                   query_crops: list,
                                   threshold: float = 0.1) -> list:
        """OWLv2 image-conditioned detection: find objects similar to query crops."""
        if not self._owlv2_model:
            raise RuntimeError("OWLv2 not loaded.")
        import torch

        image = image.convert("RGB")
        w, h = image.size

        # OWLv2 expects query_images as list of PIL images
        inputs = self._owlv2_processor(
            images=image, query_images=query_crops, return_tensors="pt"
        ).to(self._owlv2_device)

        with torch.no_grad():
            outputs = self._owlv2_model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self._owlv2_device)
        results_raw = self._owlv2_processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        dets = []
        if results_raw:
            r = results_raw[0]
            for box, score in zip(r.get("boxes", []), r.get("scores", [])):
                x1, y1, x2, y2 = [float(v) for v in box]
                dets.append({
                    "bbox": (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)),
                    "score": float(score),
                    "label": "query_match",
                })
        return dets

    def load_owlv2(self, device: str = None, progress_cb=None):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        device = device or _get_device()
        self._owlv2_device = device
        if progress_cb:
            progress_cb("Loading OWLv2...")
        self._owlv2_processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble")
        self._owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble").to(device)
        self._owlv2_model.eval()

    @staticmethod
    def _build_classification_prompt(class_rules: list) -> str:
        lines = ["Look at this image carefully. What do you see?"]
        for i, rule in enumerate(class_rules):
            lines.append(f"({LETTERS[i]}) {rule.description}")
        lines.append(f"({LETTERS[len(class_rules)]}) None of the above / something else")
        lines.append("Answer with just the letter.")
        return "\n".join(lines)

    @staticmethod
    def _parse_classification_answer(answer: str, class_rules: list):
        answer_clean = answer.strip().upper()
        # 1. Single letter
        if len(answer_clean) == 1 and answer_clean in LETTERS:
            idx = LETTERS.index(answer_clean)
            if idx < len(class_rules):
                return (class_rules[idx].class_idx, answer_clean)
            return (-1, answer_clean)  # reject option

        # 2. Pattern like (A), A., A:, A)
        match = re.search(r'\(?([A-Z])\)?[\.\s:\)]', answer_clean)
        if match:
            letter = match.group(1)
            if letter in LETTERS:
                idx = LETTERS.index(letter)
                if idx < len(class_rules):
                    return (class_rules[idx].class_idx, letter)
                return (-1, letter)

        # 3. Keyword fallback
        answer_lower = answer.lower()
        for i, rule in enumerate(class_rules):
            keywords = [w.lower() for w in rule.description.split() if len(w) > 3]
            if any(kw in answer_lower for kw in keywords):
                return (class_rules[i].class_idx, LETTERS[i])

        return (-1, "?")

    @staticmethod
    def _pad_bbox(bbox, img_size, pad_fraction):
        x1, y1, x2, y2 = bbox
        w, h = img_size
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_fraction, bh * pad_fraction
        return (
            max(0, x1 - px), max(0, y1 - py),
            min(w, x2 + px), min(h, y2 + py),
        )


# ── Smart Annotator UI ────────────────────────────────────────────────────────

class SmartAnnotatorUI:
    """Builds the 'Smart' tab and connects it to SmartAnnotator."""

    def __init__(self, tool, vlm_controller):
        self.tool = tool
        self.root = tool.root
        self.vlm = vlm_controller
        self.smart = SmartAnnotator()
        self._rule_rows = []  # list of (frame, class_var, desc_entry)
        self._running = False

    def build_smart_tab(self, notebook: ttk.Notebook):
        BG = "#f5f5f5"
        FONT = ("Segoe UI", 9)
        FONT_BOLD = ("Segoe UI", 9, "bold")

        tab = tk.Frame(notebook, bg=BG)
        notebook.add(tab, text="  Smart  ")

        # -- Scrollable content --
        canvas = tk.Canvas(tab, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=BG)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # -- Base Detection --
        tk.Label(content, text="Stage 1: Detect", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8, pady=(8, 2))

        base_row = tk.Frame(content, bg=BG)
        base_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(base_row, text="Base prompt:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.base_prompt_entry = tk.Entry(base_row, width=20, font=FONT)
        self.base_prompt_entry.insert(0, "cart")
        self.base_prompt_entry.pack(side=tk.LEFT, padx=4)

        tk.Label(content, text="(Detects ALL instances of this object)",
                 bg=BG, fg="#888", font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8)

        det_info = tk.Label(content, text="Detection model: (load in VLM tab)",
                            bg=BG, fg="#666", font=("Segoe UI", 8))
        det_info.pack(anchor=tk.W, padx=8, pady=2)
        self._det_info_label = det_info

        thresh_row = tk.Frame(content, bg=BG)
        thresh_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(thresh_row, text="Confidence:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.25)
        self.lbl_thresh = tk.Label(thresh_row, text="0.25", bg=BG, font=FONT, width=4)
        self.lbl_thresh.pack(side=tk.RIGHT)
        ttk.Scale(thresh_row, from_=0.05, to=0.95, variable=self.threshold_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_thresh.config(text=f"{float(v):.2f}")).pack(
            side=tk.RIGHT, padx=4)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Classification Rules --
        tk.Label(content, text="Stage 2: Classify (VQA)", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8)
        tk.Label(content, text="Define what each class looks like:",
                 bg=BG, fg="#666", font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8)

        self.rules_frame = tk.Frame(content, bg=BG)
        self.rules_frame.pack(fill=tk.X, padx=8, pady=4)

        # Add default rules
        self._add_rule_row(desc="a single shopping cart")
        self._add_rule_row(desc="multiple carts stacked or nested inside each other")

        rules_btn_row = tk.Frame(content, bg=BG)
        rules_btn_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(rules_btn_row, text="+ Add Rule", font=FONT, relief=tk.GROOVE,
                  bg="#dce6f0", command=lambda: self._add_rule_row()).pack(side=tk.LEFT)
        tk.Button(rules_btn_row, text="Preview Prompt", font=FONT, relief=tk.GROOVE,
                  bg="#e8e8e8", command=self._preview_prompt).pack(side=tk.LEFT, padx=4)

        # VQA model
        vqa_row = tk.Frame(content, bg=BG)
        vqa_row.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(vqa_row, text="VQA model:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.vqa_model_var = tk.StringVar(value=list(VERIFIER_MODELS.keys())[0])
        ttk.Combobox(vqa_row, textvariable=self.vqa_model_var,
                      values=list(VERIFIER_MODELS.keys()), state="readonly",
                      width=22).pack(side=tk.LEFT, padx=4)

        # Context padding
        pad_row = tk.Frame(content, bg=BG)
        pad_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(pad_row, text="Context padding:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.pad_var = tk.DoubleVar(value=0.30)
        self.lbl_pad = tk.Label(pad_row, text="0.30", bg=BG, font=FONT, width=4)
        self.lbl_pad.pack(side=tk.RIGHT)
        ttk.Scale(pad_row, from_=0.0, to=1.0, variable=self.pad_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_pad.config(text=f"{float(v):.2f}")).pack(
            side=tk.RIGHT, padx=4)
        tk.Label(content, text="(Extra area around crop for VQA context — higher = sees more)",
                 bg=BG, fg="#888", font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- OWLv2 Image Query (optional) --
        tk.Label(content, text="Stage 3: OWLv2 Image Query (optional)",
                 bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8)
        self.use_owlv2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(content, text="Use few-shot example crops as visual queries",
                         variable=self.use_owlv2_var).pack(anchor=tk.W, padx=8, pady=1)
        tk.Label(content, text="(Finds objects visually similar to your examples — load examples in few-shot bar)",
                 bg=BG, fg="#888", font=("Segoe UI", 8)).pack(anchor=tk.W, padx=8)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Scope --
        scope_frame = tk.Frame(content, bg=BG)
        scope_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(scope_frame, text="Scope:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.scope_var = tk.StringVar(value="current")
        for val, txt in [("current", "Current"), ("all", "All"),
                          ("unannotated", "New only"), ("custom", "Custom dir")]:
            ttk.Radiobutton(scope_frame, text=txt, variable=self.scope_var,
                             value=val).pack(side=tk.LEFT, padx=1)

        self.merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(content, text="Keep existing annotations",
                         variable=self.merge_var).pack(anchor=tk.W, padx=8, pady=2)

        ttk.Separator(content, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Run --
        self.btn_run = tk.Button(content, text="Run Smart Annotation", bg="#a8d5a2",
                                  font=("Segoe UI", 10, "bold"), relief=tk.GROOVE,
                                  command=self._run_thread)
        self.btn_run.pack(fill=tk.X, padx=8, pady=4)

        self.progress_bar = ttk.Progressbar(content, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=8, pady=2)
        self.lbl_status = tk.Label(content, text="Ready", bg=BG, fg="#555",
                                    font=FONT, anchor=tk.W, wraplength=240)
        self.lbl_status.pack(fill=tk.X, padx=8, pady=(0, 8))

        return tab

    # ── Rule Management ────────────────────────────────────────────────────────

    def _add_rule_row(self, desc=""):
        BG = "#f5f5f5"
        FONT = ("Segoe UI", 9)
        idx = len(self._rule_rows)

        row = tk.Frame(self.rules_frame, bg=BG)
        row.pack(fill=tk.X, pady=1)

        tk.Label(row, text=f"({LETTERS[idx]})", bg=BG, font=("Segoe UI", 9, "bold"),
                 width=3).pack(side=tk.LEFT)

        class_var = tk.StringVar()
        class_cb = ttk.Combobox(row, textvariable=class_var, state="readonly", width=12)
        classes = getattr(self.tool, "classes", [])
        class_cb["values"] = [f"{i}: {c}" for i, c in enumerate(classes)] if classes else ["No classes"]
        if class_cb["values"]:
            class_cb.current(min(idx, len(class_cb["values"]) - 1))
        class_cb.pack(side=tk.LEFT, padx=2)

        desc_entry = tk.Entry(row, width=24, font=FONT)
        desc_entry.insert(0, desc)
        desc_entry.pack(side=tk.LEFT, padx=2)

        def remove():
            row.destroy()
            self._rule_rows = [(r, v, e) for r, v, e in self._rule_rows if r != row]
            self._relabel_rules()

        tk.Button(row, text="X", font=("Segoe UI", 8), fg="red", relief=tk.FLAT,
                  command=remove).pack(side=tk.LEFT, padx=2)

        self._rule_rows.append((row, class_var, desc_entry))

    def _relabel_rules(self):
        for i, (row, _, _) in enumerate(self._rule_rows):
            for child in row.winfo_children():
                if isinstance(child, tk.Label) and child.cget("width") == 3:
                    child.config(text=f"({LETTERS[i]})")
                    break

    def _build_class_rules(self) -> list:
        rules = []
        for i, (_, class_var, desc_entry) in enumerate(self._rule_rows):
            cls_str = class_var.get()
            desc = desc_entry.get().strip()
            if not desc:
                continue
            try:
                cls_idx = int(cls_str.split(":")[0])
                cls_name = ":".join(cls_str.split(":")[1:]).strip()
            except (ValueError, IndexError):
                continue
            rules.append(ClassRule(
                class_idx=cls_idx, class_name=cls_name,
                description=desc, letter=LETTERS[i]))
        return rules

    def _preview_prompt(self):
        rules = self._build_class_rules()
        if not rules:
            messagebox.showinfo("No Rules", "Add at least one classification rule.")
            return
        prompt = SmartAnnotator._build_classification_prompt(rules)
        messagebox.showinfo("VQA Classification Prompt", prompt)

    # ── Image Collection ───────────────────────────────────────────────────────

    def _collect_target_images(self) -> list:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        scope = self.scope_var.get()

        if scope == "current":
            if not getattr(self.tool, "image_files", None) or not self.tool.images_dir:
                messagebox.showwarning("No Image", "No image loaded.")
                return None
            return [os.path.join(self.tool.images_dir,
                                 self.tool.image_files[self.tool.current_index])]
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

    # ── Run Annotation ─────────────────────────────────────────────────────────

    def _run_thread(self):
        # Validate
        if not self.vlm.annotator.is_loaded():
            messagebox.showwarning("No Detector", "Load a detection model in the VLM tab first.")
            return

        base_prompt = self.base_prompt_entry.get().strip()
        if not base_prompt:
            messagebox.showwarning("No Prompt", "Enter a base object prompt.")
            return

        class_rules = self._build_class_rules()
        if not class_rules:
            messagebox.showwarning("No Rules", "Add at least one classification rule with a description.")
            return

        vqa_model_key = self.vqa_model_var.get()

        img_paths = self._collect_target_images()
        if img_paths is None:
            return
        if not img_paths:
            messagebox.showinfo("No Images", "No images found.")
            return

        # Update detector info
        self._det_info_label.config(
            text=f"Detection model: {self.vlm.annotator._model_key or 'unknown'}")

        self._running = True
        self.btn_run.config(state=tk.DISABLED)
        self.progress_bar.configure(maximum=len(img_paths), value=0)

        threading.Thread(
            target=self._run_annotation,
            args=(img_paths, base_prompt, class_rules, vqa_model_key),
            daemon=True
        ).start()

    def _run_annotation(self, img_paths, base_prompt, class_rules, vqa_model_key):
        threshold = self.threshold_var.get()
        context_pad = self.pad_var.get()
        merge = self.merge_var.get()
        use_owlv2 = self.use_owlv2_var.get()
        is_current = self.scope_var.get() == "current"

        # Share detector from VLM tab
        self.smart.set_detector(self.vlm.annotator)

        # Load/reuse VQA classifier
        try:
            self.root.after(0, lambda: self.lbl_status.config(text="Loading VQA model..."))
            verifier = self.vlm.verifier
            needs_load = (not verifier.is_loaded()
                          or verifier._loaded_key != vqa_model_key)
            if needs_load:
                verifier.load(vqa_model_key, device=self.vlm.device_var.get())
            self.smart.set_classifier(verifier)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("VQA Load Error", str(e)))
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self._running = False
            return

        # Load OWLv2 if needed
        if use_owlv2 and not self.smart._owlv2_model:
            try:
                self.root.after(0, lambda: self.lbl_status.config(text="Loading OWLv2..."))
                self.smart.load_owlv2(device=self.vlm.device_var.get())
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("OWLv2 Error", str(e)))
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
                    self.lbl_status.config(text=f"Detecting: {os.path.basename(p)}"))

                image = Image.open(img_path).convert("RGB")
                w, h = image.size

                # Stage 1+2: Detect and classify
                results = self.smart.detect_and_classify(
                    image, base_prompt, class_rules,
                    threshold=threshold, context_pad=context_pad)

                # Stage 3: OWLv2 image queries (optional)
                if use_owlv2 and self.vlm.few_shot.total_count() > 0:
                    query_crops = []
                    query_class_idx = None
                    for label in self.vlm.few_shot.get_labels():
                        for ex in self.vlm.few_shot.examples.get(label, []):
                            if "thumbnail_b64" in ex:
                                crop = FewShotStore._b64_to_pil(ex["thumbnail_b64"])
                                query_crops.append(crop)
                                # Map to last rule's class (assumed to be the target)
                                if class_rules:
                                    query_class_idx = class_rules[-1].class_idx

                    if query_crops and query_class_idx is not None:
                        self.root.after(0, lambda p=img_path:
                            self.lbl_status.config(text=f"OWLv2 query: {os.path.basename(p)}"))
                        owlv2_dets = self.smart.detect_with_image_queries(
                            image, query_crops, threshold=0.1)
                        for det in owlv2_dets:
                            results.append({
                                "class_idx": query_class_idx,
                                "class_name": class_rules[-1].class_name,
                                "bbox": det["bbox"],
                                "score": det["score"],
                                "vqa_answer": "owlv2_query",
                                "vqa_letter": "-",
                            })

                # Write label files
                new_anns = [{"class_idx": r["class_idx"], "bbox": r["bbox"]} for r in results]

                txt_paths = set()
                txt_paths.add(os.path.splitext(img_path)[0] + ".txt")
                labels_dir = getattr(self.tool, "labels_dir", "") or ""
                if labels_dir:
                    basename = os.path.basename(img_path)
                    txt_paths.add(os.path.join(
                        labels_dir, os.path.splitext(basename)[0] + ".txt"))

                bbox_lines = []
                for ann in new_anns:
                    x1, y1, x2, y2 = ann["bbox"]
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    xc, yc = (x1 / w) + bw / 2, (y1 / h) + bh / 2
                    bbox_lines.append(
                        f"{ann['class_idx']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                for txt_path in txt_paths:
                    existing = []
                    if merge and os.path.exists(txt_path):
                        with open(txt_path, "r") as f:
                            existing = f.readlines()
                    if existing or bbox_lines:
                        with open(txt_path, "w") as f:
                            f.writelines(existing)
                            f.writelines(bbox_lines)

                total_boxes += len(new_anns)
                if new_anns:
                    annotated += 1
                    if is_current:
                        current_img_anns = new_anns

            except Exception as e:
                print(f"Smart annotation error on {img_path}: {e}")
                errors += 1

            self.root.after(0, lambda v=i+1: self.progress_bar.configure(value=v))

        def on_done():
            self._running = False
            self.btn_run.config(state=tk.NORMAL)
            msg = f"Done: {annotated}/{total} images, {total_boxes} boxes classified"
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
            if getattr(self.tool, "image_files", None):
                self.tool.load_current_image()

            messagebox.showinfo("Smart Annotation Complete", msg)

        self.root.after(0, on_done)


# ── Entry Point ─────────────────────────────────────────────────────────────────

def create_smart_ui(tool, vlm_controller) -> SmartAnnotatorUI:
    return SmartAnnotatorUI(tool, vlm_controller)
