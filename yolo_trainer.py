"""
One-click YOLO Training tab for the annotation tool.

Given annotated images (YOLO format .txt labels), this module:
  1. Splits data into train/val sets
  2. Generates a YOLO dataset YAML
  3. Trains a YOLOv8/v11/v12 model via ultralytics
  4. Shows live training progress

Usage:
    from yolo_trainer import YOLOTrainerUI
    trainer_ui = YOLOTrainerUI(tool)
    trainer_ui.build_train_tab(notebook)
"""

import os
import shutil
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


YOLO_BASE_MODELS = {
    # ── YOLOv5 (legacy, widely used) ──
    "YOLOv5n (nano)": "yolov5nu.pt",
    "YOLOv5s (small)": "yolov5su.pt",
    "YOLOv5m (medium)": "yolov5mu.pt",
    "YOLOv5l (large)": "yolov5lu.pt",
    # ── YOLOv6 ──
    "YOLOv6n (nano)": "yolov6-3.0-n.pt",
    "YOLOv6s (small)": "yolov6-3.0-s.pt",
    "YOLOv6m (medium)": "yolov6-3.0-m.pt",
    "YOLOv6l (large)": "yolov6-3.0-l.pt",
    # ── YOLOv8 ──
    "YOLOv8n (nano)": "yolov8n.pt",
    "YOLOv8s (small)": "yolov8s.pt",
    "YOLOv8m (medium)": "yolov8m.pt",
    "YOLOv8l (large)": "yolov8l.pt",
    "YOLOv8x (extra-large)": "yolov8x.pt",
    # ── YOLOv11 ──
    "YOLOv11n (nano)": "yolo11n.pt",
    "YOLOv11s (small)": "yolo11s.pt",
    "YOLOv11m (medium)": "yolo11m.pt",
    "YOLOv11l (large)": "yolo11l.pt",
    "YOLOv11x (extra-large)": "yolo11x.pt",
    # ── YOLOv12 ──
    "YOLOv12n (nano)": "yolo12n.pt",
    "YOLOv12s (small)": "yolo12s.pt",
    "YOLOv12m (medium)": "yolo12m.pt",
    "YOLOv12l (large)": "yolo12l.pt",
    # ── YOLOv26 ──
    "YOLOv26n (nano)": "yolo26n.pt",
    "YOLOv26s (small)": "yolo26s.pt",
    "YOLOv26m (medium)": "yolo26m.pt",
    "YOLOv26l (large)": "yolo26l.pt",
    "YOLOv26x (extra-large)": "yolo26x.pt",
    # ── YOLOE (open-vocab) ──
    "YOLOE-v8s": "yolov8s-worldv2.pt",
}

# Augmentation presets passed to model.train()
_AUG_PRESETS = {
    "Default": {},  # ultralytics defaults
    "Light": dict(degrees=0, translate=0.1, scale=0.2, flipud=0.0,
                  mosaic=0.5, mixup=0.0),
    "None": dict(degrees=0, translate=0, scale=0, flipud=0, fliplr=0,
                 mosaic=0, mixup=0, hsv_h=0, hsv_s=0, hsv_v=0),
}


def _fast_copy(src, dst):
    """Try hardlink first (instant, no disk space), fall back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class YOLOTrainerUI:
    """Builds the 'Train' tab for one-click YOLO fine-tuning."""

    def __init__(self, tool):
        self.tool = tool
        self.root = tool.root
        self._training = False
        self._train_thread = None

    def build_train_tab(self, notebook: ttk.Notebook):
        BG = "#f5f5f5"
        FONT = ("Segoe UI", 9)
        FONT_BOLD = ("Segoe UI", 9, "bold")
        FONT_SM = ("Segoe UI", 8)

        outer = tk.Frame(notebook, bg=BG)
        notebook.add(outer, text="  Train  ")

        # Scrollable canvas wrapper
        _canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        _vsb = ttk.Scrollbar(outer, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tab = tk.Frame(_canvas, bg=BG)
        _win = _canvas.create_window((0, 0), window=tab, anchor="nw")
        tab.bind("<Configure>", lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.bind("<Configure>", lambda e: _canvas.itemconfig(_win, width=e.width))

        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        _canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        # -- Data Source --
        tk.Label(tab, text="Data Source", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8, pady=(8, 2))

        tk.Label(tab, text="Uses the currently loaded workspace images + labels.",
                 bg=BG, fg="#666", font=FONT_SM).pack(anchor=tk.W, padx=8)

        self.lbl_data_info = tk.Label(tab, text="No data loaded", bg=BG, fg="#888",
                                       font=FONT)
        self.lbl_data_info.pack(anchor=tk.W, padx=8, pady=2)

        tk.Button(tab, text="Refresh Data Count", font=FONT, relief=tk.GROOVE,
                  bg="#dce6f0", command=self._refresh_data_info).pack(
            anchor=tk.W, padx=8, pady=2)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Train/Val Split --
        tk.Label(tab, text="Train / Val Split", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8)

        split_row = tk.Frame(tab, bg=BG)
        split_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(split_row, text="Val fraction:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.val_split_var = tk.DoubleVar(value=0.15)
        self.lbl_split = tk.Label(split_row, text="0.15", bg=BG, font=FONT, width=4)
        self.lbl_split.pack(side=tk.RIGHT)
        ttk.Scale(split_row, from_=0.05, to=0.50, variable=self.val_split_var,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self.lbl_split.config(text=f"{float(v):.2f}")).pack(
            side=tk.RIGHT, padx=4)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Model --
        tk.Label(tab, text="Base Model", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8)

        model_row = tk.Frame(tab, bg=BG)
        model_row.pack(fill=tk.X, padx=8, pady=2)
        self.base_model_var = tk.StringVar(value=list(YOLO_BASE_MODELS.keys())[0])
        ttk.Combobox(model_row, textvariable=self.base_model_var,
                      values=list(YOLO_BASE_MODELS.keys()), state="readonly",
                      width=22).pack(side=tk.LEFT)

        # Or custom .pt
        custom_row = tk.Frame(tab, bg=BG)
        custom_row.pack(fill=tk.X, padx=8, pady=2)
        self.use_custom_model_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(custom_row, text="Use custom .pt:",
                         variable=self.use_custom_model_var).pack(side=tk.LEFT)
        self.custom_model_path_var = tk.StringVar()
        tk.Entry(custom_row, textvariable=self.custom_model_path_var, width=14,
                 font=FONT).pack(side=tk.LEFT, padx=4)
        tk.Button(custom_row, text="...", font=FONT, relief=tk.GROOVE,
                  command=self._browse_custom_model).pack(side=tk.LEFT)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Training Params --
        tk.Label(tab, text="Training Parameters", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8)

        def _param_row(parent, label_text, var, width=6):
            row = tk.Frame(parent, bg=BG)
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(row, text=label_text, bg=BG, font=FONT).pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=width, font=FONT).pack(
                side=tk.LEFT, padx=4)
            return row

        # Epochs
        self.epochs_var = tk.IntVar(value=50)
        _param_row(tab, "Epochs:", self.epochs_var)

        # Batch size
        self.batch_var = tk.IntVar(value=16)
        _param_row(tab, "Batch size:", self.batch_var)

        # Image size
        imgsz_row = tk.Frame(tab, bg=BG)
        imgsz_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(imgsz_row, text="Image size:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.imgsz_var = tk.IntVar(value=640)
        ttk.Combobox(imgsz_row, textvariable=self.imgsz_var,
                      values=[320, 416, 512, 640, 800, 1024, 1280],
                      width=6).pack(side=tk.LEFT, padx=4)

        # Learning rate
        self.lr0_var = tk.DoubleVar(value=0.01)
        _param_row(tab, "Learning rate (lr0):", self.lr0_var, width=8)

        # Final LR fraction
        self.lrf_var = tk.DoubleVar(value=0.01)
        _param_row(tab, "Final LR fraction (lrf):", self.lrf_var, width=8)

        # Warmup epochs
        self.warmup_var = tk.DoubleVar(value=3.0)
        _param_row(tab, "Warmup epochs:", self.warmup_var, width=6)

        # Weight decay
        self.weight_decay_var = tk.DoubleVar(value=0.0005)
        _param_row(tab, "Weight decay:", self.weight_decay_var, width=8)

        # Optimizer
        opt_row = tk.Frame(tab, bg=BG)
        opt_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(opt_row, text="Optimizer:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.optimizer_var = tk.StringVar(value="auto")
        ttk.Combobox(opt_row, textvariable=self.optimizer_var,
                      values=["auto", "SGD", "Adam", "AdamW"],
                      state="readonly", width=8).pack(side=tk.LEFT, padx=4)

        # Freeze layers
        self.freeze_var = tk.IntVar(value=0)
        fr = _param_row(tab, "Freeze layers:", self.freeze_var, width=4)
        tk.Label(fr, text="(backbone)", bg=BG, fg="#888", font=FONT_SM).pack(side=tk.LEFT)

        # Patience (early stopping)
        self.patience_var = tk.IntVar(value=20)
        pr = _param_row(tab, "Patience:", self.patience_var, width=4)
        tk.Label(pr, text="(0=off)", bg=BG, fg="#888", font=FONT_SM).pack(side=tk.LEFT)

        # Augmentation preset
        aug_row = tk.Frame(tab, bg=BG)
        aug_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(aug_row, text="Augmentation:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.aug_var = tk.StringVar(value="Default")
        ttk.Combobox(aug_row, textvariable=self.aug_var,
                      values=list(_AUG_PRESETS.keys()),
                      state="readonly", width=10).pack(side=tk.LEFT, padx=4)

        # Device
        dev_row = tk.Frame(tab, bg=BG)
        dev_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(dev_row, text="Device:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.device_var = tk.StringVar(value=self._detect_device())
        ttk.Combobox(dev_row, textvariable=self.device_var,
                      values=["cpu", "0", "0,1"], state="readonly",
                      width=6).pack(side=tk.LEFT, padx=4)

        # Output dir
        out_row = tk.Frame(tab, bg=BG)
        out_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(out_row, text="Output:", bg=BG, font=FONT).pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar()
        tk.Entry(out_row, textvariable=self.output_dir_var, width=16, font=FONT).pack(
            side=tk.LEFT, padx=4)
        tk.Button(out_row, text="...", font=FONT, relief=tk.GROOVE,
                  command=self._browse_output_dir).pack(side=tk.LEFT)

        ttk.Separator(tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # -- Run --
        self.btn_train = tk.Button(tab, text="Start Training", bg="#a8d5a2",
                                    font=("Segoe UI", 10, "bold"), relief=tk.GROOVE,
                                    command=self._start_training)
        self.btn_train.pack(fill=tk.X, padx=8, pady=4)

        self.btn_stop = tk.Button(tab, text="Stop Training", bg="#f0d5d5",
                                   font=FONT, relief=tk.GROOVE, state=tk.DISABLED,
                                   command=self._stop_training)
        self.btn_stop.pack(fill=tk.X, padx=8, pady=2)

        self.progress_bar = ttk.Progressbar(tab, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=8, pady=2)

        self.lbl_status = tk.Label(tab, text="Ready", bg=BG, fg="#555",
                                    font=FONT, anchor=tk.W, wraplength=240)
        self.lbl_status.pack(fill=tk.X, padx=8, pady=(0, 2))

        # -- Training Log --
        tk.Label(tab, text="Training Log", bg=BG, font=FONT_BOLD).pack(
            anchor=tk.W, padx=8, pady=(4, 2))
        log_frame = tk.Frame(tab, bg=BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, height=10, font=("Consolas", 8),
                                 bg="#1e1e1e", fg="#d4d4d4", relief=tk.GROOVE,
                                 bd=1, wrap=tk.WORD, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                    command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure log text tags for color coding
        self.log_text.tag_configure("header", foreground="#569cd6", font=("Consolas", 8, "bold"))
        self.log_text.tag_configure("epoch", foreground="#4ec9b0")
        self.log_text.tag_configure("metric", foreground="#dcdcaa")
        self.log_text.tag_configure("best", foreground="#6a9955", font=("Consolas", 8, "bold"))
        self.log_text.tag_configure("warn", foreground="#ce9178")

        return outer

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _log(self, text, tag=None):
        """Append a line to the training log (thread-safe via root.after)."""
        def _append():
            self.log_text.config(state=tk.NORMAL)
            if tag:
                self.log_text.insert(tk.END, text + "\n", tag)
            else:
                self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _append)

    @staticmethod
    def _detect_device():
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except ImportError:
            pass
        return "cpu"

    def _browse_custom_model(self):
        p = filedialog.askopenfilename(
            title="Select YOLO .pt Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if p:
            self.custom_model_path_var.set(p)

    def _browse_output_dir(self):
        d = filedialog.askdirectory(title="Select Training Output Directory")
        if d:
            self.output_dir_var.set(d)

    def _refresh_data_info(self):
        images_dir = getattr(self.tool, "images_dir", "") or ""
        classes = getattr(self.tool, "classes", [])
        if not images_dir:
            self.lbl_data_info.config(text="No images loaded", fg="#c00")
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        total_images = 0
        annotated = 0
        labels_dir = getattr(self.tool, "labels_dir", "") or ""
        for root_d, _, files in os.walk(images_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    total_images += 1
                    base = os.path.splitext(f)[0]
                    txt = os.path.join(root_d, base + ".txt")
                    if os.path.exists(txt):
                        annotated += 1
                    elif labels_dir and labels_dir != images_dir:
                        if os.path.exists(os.path.join(labels_dir, base + ".txt")):
                            annotated += 1

        self.lbl_data_info.config(
            text=f"{annotated}/{total_images} annotated | {len(classes)} classes",
            fg="#333" if annotated > 0 else "#c00")

        # Auto-set output dir if empty
        if not self.output_dir_var.get():
            self.output_dir_var.set(os.path.join(images_dir, "..", "yolo_training"))

    # ── Dataset Preparation ───────────────────────────────────────────────────

    def _prepare_dataset(self, output_dir, val_fraction):
        """Create YOLO dataset structure with hardlinks (fast) or file copy (fallback)."""
        images_dir = self.tool.images_dir
        labels_dir = getattr(self.tool, "labels_dir", "") or images_dir
        classes = self.tool.classes

        # Collect annotated images
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        pairs = []
        for root_d, _, files in os.walk(images_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() not in valid_exts:
                    continue
                img_path = os.path.join(root_d, f)
                base = os.path.splitext(f)[0]
                lbl_path = os.path.join(root_d, base + ".txt")
                if not os.path.exists(lbl_path) and labels_dir:
                    lbl_path = os.path.join(labels_dir, base + ".txt")
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    pairs.append((img_path, lbl_path))

        if not pairs:
            raise ValueError("No annotated images found. Annotate some images first.")

        if len(pairs) < 3:
            raise ValueError(f"Only {len(pairs)} annotated images found. Need at least 3.")

        random.shuffle(pairs)
        n_val = max(2, int(len(pairs) * val_fraction))  # at least 2 val images
        # Ensure enough for training too
        if len(pairs) - n_val < 2:
            n_val = max(1, len(pairs) - 2)
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        if not train_pairs:
            raise ValueError("Not enough images for training after split.")

        # Create directory structure
        dataset_dir = os.path.join(output_dir, "dataset")
        for split in ("train", "val"):
            os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

        # Copy files (use shutil.copy2 for reliability — hardlinks fail cross-drive on Windows)
        for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
            for img_path, lbl_path in split_pairs:
                fname = os.path.basename(img_path)
                lbl_fname = os.path.splitext(fname)[0] + ".txt"
                dst_img = os.path.join(dataset_dir, "images", split, fname)
                dst_lbl = os.path.join(dataset_dir, "labels", split, lbl_fname)
                for p in (dst_img, dst_lbl):
                    if os.path.exists(p):
                        os.remove(p)
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

        # Write data.yaml with forward slashes (YOLO requires POSIX paths)
        dataset_dir_posix = dataset_dir.replace("\\", "/")
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {dataset_dir_posix}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {classes}\n")

        # Write classes.txt alongside the dataset
        with open(os.path.join(dataset_dir, "classes.txt"), "w") as f:
            for cls in classes:
                f.write(cls + "\n")

        return yaml_path, len(train_pairs), len(val_pairs)

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self):
        # Validate
        images_dir = getattr(self.tool, "images_dir", "") or ""
        classes = getattr(self.tool, "classes", [])
        if not images_dir:
            messagebox.showwarning("No Data", "Load images first.")
            return
        if not classes:
            messagebox.showwarning("No Classes", "Add or load classes first.")
            return

        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            messagebox.showerror("Missing Package",
                "ultralytics is required.\n\npip install ultralytics")
            return

        # Get model path
        if self.use_custom_model_var.get():
            model_path = self.custom_model_path_var.get().strip()
            if not model_path or not os.path.exists(model_path):
                messagebox.showwarning("Invalid Model", "Select a valid custom .pt file.")
                return
        else:
            model_key = self.base_model_var.get()
            model_path = YOLO_BASE_MODELS.get(model_key, "yolov8n.pt")

        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            output_dir = os.path.join(images_dir, "..", "yolo_training")
            self.output_dir_var.set(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Prepare dataset
        try:
            self.lbl_status.config(text="Preparing dataset...")
            self.root.update()
            yaml_path, n_train, n_val = self._prepare_dataset(
                output_dir, self.val_split_var.get())
        except Exception as e:
            messagebox.showerror("Dataset Error", str(e))
            return

        self.lbl_status.config(text=f"Dataset ready: {n_train} train, {n_val} val")

        # Read all hyperparameters from UI
        epochs = self.epochs_var.get()
        batch = self.batch_var.get()
        imgsz = self.imgsz_var.get()
        device = self.device_var.get()
        lr0 = self.lr0_var.get()
        lrf = self.lrf_var.get()
        warmup_epochs = self.warmup_var.get()
        weight_decay = self.weight_decay_var.get()
        optimizer = self.optimizer_var.get()
        freeze = self.freeze_var.get()
        patience = self.patience_var.get()
        aug_preset = self.aug_var.get()

        self._training = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_bar.configure(maximum=epochs, value=0)

        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

        def train_worker():
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
                best_mAP = [0.0]

                self._log(f"{'='*50}", "header")
                self._log(f"  Training: {model_path}", "header")
                self._log(f"  Dataset: {n_train} train / {n_val} val", "header")
                self._log(f"  Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}", "header")
                self._log(f"  LR: {lr0} | Optimizer: {optimizer} | Device: {device}", "header")
                self._log(f"{'='*50}", "header")
                self._log("")

                # After each training epoch — log losses
                def on_train_epoch_end(trainer):
                    epoch = trainer.epoch + 1
                    # Get training losses from trainer.loss_items
                    lr_val = "?"
                    try:
                        lr_val = f"{trainer.optimizer.param_groups[0]['lr']:.6f}"
                    except Exception:
                        pass

                    loss_str = ""
                    try:
                        if trainer.loss_items is not None:
                            loss_names = trainer.loss_names if hasattr(trainer, 'loss_names') else []
                            items = trainer.loss_items.cpu().tolist() if hasattr(trainer.loss_items, 'cpu') else []
                            if loss_names and items:
                                parts = [f"{n}={v:.4f}" for n, v in zip(loss_names, items)]
                                loss_str = " | ".join(parts)
                            elif items:
                                loss_str = f"total={sum(items):.4f}"
                    except Exception:
                        pass

                    line = f"[Epoch {epoch:3d}/{epochs}] lr={lr_val}"
                    if loss_str:
                        line += f" | {loss_str}"
                    self._log(line, "epoch")

                    self.root.after(0, lambda e=epoch, ls=loss_str, l=lr_val: (
                        self.progress_bar.configure(value=e),
                        self.lbl_status.config(
                            text=f"Epoch {e}/{epochs} | {ls} | lr={l}")
                    ))

                # After validation — log mAP and val metrics
                def on_fit_epoch_end(trainer):
                    epoch = trainer.epoch + 1
                    metrics = trainer.metrics or {}
                    if not metrics:
                        return

                    mAP50 = metrics.get("metrics/mAP50(B)", None)
                    mAP50_95 = metrics.get("metrics/mAP50-95(B)", None)
                    val_box = metrics.get("val/box_loss", None)
                    val_cls = metrics.get("val/cls_loss", None)
                    precision = metrics.get("metrics/precision(B)", None)
                    recall = metrics.get("metrics/recall(B)", None)

                    parts = []
                    if val_box is not None:
                        parts.append(f"val_box={val_box:.4f}")
                    if val_cls is not None:
                        parts.append(f"val_cls={val_cls:.4f}")
                    if precision is not None:
                        parts.append(f"P={precision:.3f}")
                    if recall is not None:
                        parts.append(f"R={recall:.3f}")
                    if mAP50 is not None:
                        parts.append(f"mAP50={mAP50:.3f}")
                    if mAP50_95 is not None:
                        parts.append(f"mAP50-95={mAP50_95:.3f}")

                    if parts:
                        val_line = f"         val: {' | '.join(parts)}"
                        is_best = mAP50 is not None and mAP50 > best_mAP[0]
                        if is_best:
                            best_mAP[0] = mAP50
                            val_line += "  ★ best"
                            self._log(val_line, "best")
                        else:
                            self._log(val_line, "metric")

                    # Update status with mAP
                    if mAP50 is not None:
                        self.root.after(0, lambda e=epoch, m=mAP50, m95=mAP50_95 or 0: (
                            self.lbl_status.config(
                                text=f"Epoch {e}/{epochs} | mAP50={m:.3f} | mAP50-95={m95:.3f}")
                        ))

                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

                # Build training kwargs
                train_kwargs = dict(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    device=device,
                    lr0=lr0,
                    lrf=lrf,
                    warmup_epochs=warmup_epochs,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    patience=patience,
                    project=output_dir,
                    name="train",
                    exist_ok=True,
                    verbose=True,
                    workers=0 if os.name == "nt" else 4,  # Windows dataloader fix
                )

                # Freeze backbone layers if requested
                if freeze > 0:
                    train_kwargs["freeze"] = freeze

                # Add augmentation preset
                aug_params = _AUG_PRESETS.get(aug_preset, {})
                train_kwargs.update(aug_params)

                results = model.train(**train_kwargs)

                # Training complete
                best_pt = os.path.join(output_dir, "train", "weights", "best.pt")
                last_pt = os.path.join(output_dir, "train", "weights", "last.pt")
                best_path = best_pt if os.path.exists(best_pt) else last_pt

                self._log("")
                self._log(f"{'='*50}", "header")
                self._log(f"  Training complete!", "best")
                self._log(f"  Best mAP50: {best_mAP[0]:.3f}", "best")
                self._log(f"  Model saved: {best_path}", "header")
                self._log(f"{'='*50}", "header")

                def on_done():
                    self._training = False
                    self.btn_train.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                    self.progress_bar.configure(value=epochs)
                    self.lbl_status.config(
                        text=f"Complete! Best mAP50={best_mAP[0]:.3f}")
                    messagebox.showinfo("Training Complete",
                        f"Model saved to:\n{best_path}\n\n"
                        f"Best mAP50: {best_mAP[0]:.3f}\n"
                        f"Train: {n_train} images | Val: {n_val} images\n"
                        f"Epochs: {epochs} | LR: {lr0} | Optimizer: {optimizer}")

                self.root.after(0, on_done)

            except Exception as e:
                self._log(f"ERROR: {e}", "warn")

                def on_error():
                    self._training = False
                    self.btn_train.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                    self.lbl_status.config(text=f"Error: {str(e)[:80]}")
                    messagebox.showerror("Training Error", str(e))

                self.root.after(0, on_error)

        self._train_thread = threading.Thread(target=train_worker, daemon=True)
        self._train_thread.start()

    def _stop_training(self):
        self._training = False
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Stopping after current epoch...")
        messagebox.showinfo("Stopping",
            "Training will stop after the current epoch completes.\n"
            "The best model so far will be saved.")
