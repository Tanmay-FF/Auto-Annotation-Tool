import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import threading
import glob
import os

try:
    from detection_labelling_auto_annotate import create_vlm_controller
    _VLM_AVAILABLE = True
except ImportError:
    _VLM_AVAILABLE = False

class ClassMappingDialog(tk.Toplevel):
    """Dialog for mapping one or more YOLO model classes to workspace classes.

    ``self.result`` is a list of ``(model_cls_idx, workspace_cls_idx)`` tuples,
    or ``None`` if the user cancelled.  The old single-mapping API
    ``(model_idx, ws_idx)`` is preserved when exactly one mapping is added.
    """

    def __init__(self, parent, model_names, workspace_classes):
        super().__init__(parent)
        self.title("Class Mapping")
        self.geometry("500x420")
        self.transient(parent)
        self.grab_set()

        self.result = None
        self._model_names = model_names
        self._ws_classes = workspace_classes
        self._mappings = []  # list of (model_idx, ws_idx)

        FONT = ("Segoe UI", 9)

        # --- Add mapping row ---
        add_frame = tk.LabelFrame(self, text="Add mapping", font=("Segoe UI", 9, "bold"), padx=8, pady=6)
        add_frame.pack(fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(add_frame, text="Model class:", font=FONT).grid(row=0, column=0, sticky=tk.W)
        self.model_class_var = tk.StringVar(self)
        self.model_class_cb = ttk.Combobox(add_frame, textvariable=self.model_class_var,
                                            state="readonly", width=30)
        self.model_class_cb['values'] = [f"{k}: {v}" for k, v in model_names.items()]
        if self.model_class_cb['values']:
            self.model_class_cb.current(0)
        self.model_class_cb.grid(row=0, column=1, padx=4, pady=2)

        tk.Label(add_frame, text="Map to:", font=FONT).grid(row=1, column=0, sticky=tk.W)
        self.ws_class_var = tk.StringVar(self)
        self.ws_class_cb = ttk.Combobox(add_frame, textvariable=self.ws_class_var,
                                         state="readonly", width=30)
        self.ws_class_cb['values'] = [f"{i}: {c}" for i, c in enumerate(workspace_classes)]
        if self.ws_class_cb['values']:
            self.ws_class_cb.current(0)
        self.ws_class_cb.grid(row=1, column=1, padx=4, pady=2)

        tk.Button(add_frame, text="+ Add", font=FONT, bg="#dce6f0", relief=tk.GROOVE,
                  width=8, command=self._add_mapping).grid(row=0, column=2, rowspan=2, padx=(8, 0))

        # --- Mappings list ---
        list_frame = tk.LabelFrame(self, text="Mappings (model → workspace)", font=("Segoe UI", 9, "bold"),
                                    padx=8, pady=6)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.mapping_listbox = tk.Listbox(list_frame, height=6, font=FONT, relief=tk.GROOVE, bd=1)
        self.mapping_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        tk.Button(list_frame, text="Remove Selected", font=FONT, bg="#f0d5d5", relief=tk.GROOVE,
                  command=self._remove_mapping).pack(anchor=tk.W)

        tk.Label(self, text="Unmapped model predictions will be ignored.", fg="gray",
                 font=("Segoe UI", 8)).pack(pady=2)

        # --- Bottom buttons ---
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Start Annotation", width=15, bg="lightblue",
                  font=FONT, command=self.on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", width=15, font=FONT,
                  command=self.on_cancel).pack(side=tk.LEFT, padx=10)

        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def _add_mapping(self):
        mc = self.model_class_cb.get()
        wc = self.ws_class_cb.get()
        if not mc or not wc:
            return
        model_idx = int(mc.split(":")[0])
        ws_idx = int(wc.split(":")[0])
        # Avoid duplicate model class mapping
        for existing_m, _ in self._mappings:
            if existing_m == model_idx:
                messagebox.showwarning("Duplicate", f"Model class {mc} is already mapped.")
                return
        self._mappings.append((model_idx, ws_idx))
        model_name = self._model_names.get(model_idx, str(model_idx))
        ws_name = self._ws_classes[ws_idx] if ws_idx < len(self._ws_classes) else str(ws_idx)
        self.mapping_listbox.insert(tk.END, f"{model_idx}: {model_name}  →  {ws_idx}: {ws_name}")

    def _remove_mapping(self):
        sel = self.mapping_listbox.curselection()
        if sel:
            idx = sel[0]
            self.mapping_listbox.delete(idx)
            del self._mappings[idx]

    def on_ok(self):
        if not self._mappings:
            messagebox.showwarning("No Mappings", "Add at least one class mapping.")
            return
        self.result = list(self._mappings)
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", 
    "yellow", "brown", "pink", "lime", "teal", "navy", "maroon", "olive"
]

class YOLOAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Tool")
        self.root.geometry("1000x700")
        
        # Application state
        self.images_dir = ""
        self.labels_dir = ""
        self.classes_file = ""
        
        self.image_files = []
        self.current_index = 0
        
        self.classes = []
        self.current_class_idx = 0
        
        # Bounding boxes for the current image:
        # dict of { 'class_idx': int, 'bbox': (x_min, y_min, x_max, y_max) } in original image coords
        self.annotations = [] 
        
        # Display state
        self.current_image = None
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        self.scale_f = 1.0
        self.x_offset = 0
        self.y_offset = 0
        
        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect_id = None
        self.selected_rect_idx = None
        self.resize_handle_idx = None # 0=TL, 1=TR, 2=BL, 3=BR
        self.moving_rect = False
        self.move_start_x = None
        self.move_start_y = None
        self.orig_bbox = None
        self.canvas_rects = [] # stores canvas graphic IDs
        self.canvas_texts = []
        
        self.crosshair_h = None
        self.crosshair_v = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # ── Fonts ──
        FONT = ("Segoe UI", 10)
        FONT_BOLD = ("Segoe UI", 10, "bold")
        FONT_SM = ("Segoe UI", 9)
        FONT_SM_BOLD = ("Segoe UI", 9, "bold")
        BG = "#f5f5f5"
        BG_ACCENT = "#e8edf2"
        BTN_PAD = 4

        # Resizable paned layout: left panel + canvas
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6,
                                     sashrelief=tk.RAISED, bg="#c0c0c0")
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Controls)
        left_panel = tk.Frame(self.paned, bg=BG)
        self.paned.add(left_panel, minsize=250, width=300)

        # -- Top: file buttons (always visible) --
        tk.Label(left_panel, text="Workspace", bg=BG, font=FONT_BOLD).pack(anchor=tk.W, padx=8, pady=(6, 2))

        file_frame = tk.Frame(left_panel, bg=BG)
        file_frame.pack(fill=tk.X, padx=8)

        tk.Button(file_frame, text="Open Images", command=self.load_images_dir,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.LEFT, fill=tk.X, expand=True, pady=BTN_PAD, padx=(0, 2))
        tk.Button(file_frame, text="Open Labels", command=self.load_labels_dir,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.LEFT, fill=tk.X, expand=True, pady=BTN_PAD, padx=(2, 2))
        tk.Button(file_frame, text="Open Video", command=self.load_video,
                  font=FONT_SM, relief=tk.GROOVE, bg="#e0d8f0").pack(side=tk.LEFT, fill=tk.X, expand=True, pady=BTN_PAD, padx=(2, 0))

        dir_info = tk.Frame(left_panel, bg=BG)
        dir_info.pack(fill=tk.X, padx=8)
        self.lbl_images_dir = tk.Label(dir_info, text="Images: —", fg="#666", bg=BG, font=FONT_SM, anchor=tk.W)
        self.lbl_images_dir.pack(fill=tk.X)
        self.lbl_labels_dir = tk.Label(dir_info, text="Labels: —", fg="#666", bg=BG, font=FONT_SM, anchor=tk.W)
        self.lbl_labels_dir.pack(fill=tk.X)

        tk.Button(left_panel, text="Load classes.txt", command=self.load_classes_file,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(fill=tk.X, padx=8, pady=BTN_PAD)

        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # -- Bottom: Navigation (pack bottom-up so they stick to bottom) --
        nav_frame = tk.Frame(left_panel, bg=BG)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(6, 4))

        tk.Button(nav_frame, text="< Prev (A)", command=self.prev_image,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(nav_frame, text="Next (D) >", command=self.next_image,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        self.lbl_progress = tk.Label(nav_frame, text="0 / 0", bg=BG, font=FONT_BOLD)
        self.lbl_progress.pack(pady=2)

        jump_frame = tk.Frame(left_panel, bg=BG)
        jump_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=2)
        tk.Label(jump_frame, text="Jump to:", bg=BG, font=FONT_SM).pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(jump_frame, width=8, font=FONT_SM)
        self.jump_entry.pack(side=tk.LEFT, padx=4)
        self.jump_entry.bind('<Return>', self.jump_to_image)
        tk.Button(jump_frame, text="Go", command=self.jump_to_image,
                  font=FONT_SM, relief=tk.GROOVE, width=4).pack(side=tk.LEFT)

        tk.Button(left_panel, text="Save  (Ctrl+S)", command=self.save_annotations,
                  bg="#a8d5a2", font=FONT_BOLD, relief=tk.GROOVE).pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # -- VLM controller (few-shot bar moved inside VLM tab) --
        self.vlm_controller = None
        if _VLM_AVAILABLE:
            self.vlm_controller = create_vlm_controller(self)

        # -- Notebook with Annotate / VLM tabs --
        style = ttk.Style()
        style.configure("Panel.TNotebook.Tab", font=FONT_BOLD, padding=[12, 4])
        self.left_notebook = ttk.Notebook(left_panel, style="Panel.TNotebook")
        self.left_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        # === Annotate Tab ===
        annotate_tab = tk.Frame(self.left_notebook, bg=BG)
        self.left_notebook.add(annotate_tab, text="  Annotate  ")

        tk.Button(annotate_tab, text="Auto Annotate (YOLO)...", command=self.auto_annotate,
                  bg="#b8d4f0", font=FONT_SM, relief=tk.GROOVE).pack(fill=tk.X, padx=8, pady=(8, 4))

        export_row = tk.Frame(annotate_tab, bg=BG)
        export_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(export_row, text="Export All CSV", command=lambda: self.export_annotations(False),
                  bg="#f5edb8", font=FONT_SM, relief=tk.GROOVE).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(export_row, text="Export To Current", command=lambda: self.export_annotations(True),
                  bg="#f5edb8", font=FONT_SM, relief=tk.GROOVE).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Separator(annotate_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Classes Listbox
        tk.Label(annotate_tab, text="Classes", bg=BG, font=FONT_SM_BOLD).pack(anchor=tk.W, padx=8)
        self.listbox_classes = tk.Listbox(annotate_tab, height=5, exportselection=False,
                                           font=FONT_SM, relief=tk.GROOVE, bd=1)
        self.listbox_classes.pack(fill=tk.X, padx=8, pady=2)
        self.listbox_classes.bind('<<ListboxSelect>>', self.on_class_select)

        cls_btn_row = tk.Frame(annotate_tab, bg=BG)
        cls_btn_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(cls_btn_row, text="Add Class", command=self.add_class,
                  font=FONT_SM, relief=tk.GROOVE, bg="#dce6f0").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(cls_btn_row, text="Remove Class", command=self.remove_class,
                  font=FONT_SM, relief=tk.GROOVE, bg="#f0d5d5").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Separator(annotate_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # Current Annotations Listbox
        tk.Label(annotate_tab, text="Annotations  (click to select)", bg=BG, font=FONT_SM_BOLD).pack(anchor=tk.W, padx=8)
        self.listbox_annotations = tk.Listbox(annotate_tab, height=5, exportselection=False,
                                               font=FONT_SM, relief=tk.GROOVE, bd=1)
        self.listbox_annotations.pack(fill=tk.X, padx=8, pady=2)
        self.listbox_annotations.bind('<<ListboxSelect>>', self.on_annotation_select)

        tk.Button(annotate_tab, text="Delete Selected  (Del)", command=self.delete_selected,
                  font=FONT_SM, relief=tk.GROOVE, bg="#f0d5d5").pack(fill=tk.X, padx=8, pady=2)

        discard_row = tk.Frame(annotate_tab, bg=BG)
        discard_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(discard_row, text="Discard Current", command=self.discard_current_annotations,
                  font=FONT_SM, relief=tk.GROOVE, bg="#f0c5c5").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(discard_row, text="Discard All", command=self.discard_all_annotations,
                  font=FONT_SM, relief=tk.GROOVE, bg="#e8a8a8").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Label(annotate_tab, text="Drag edges to resize  |  Drag inside to move",
                 bg=BG, fg="#888", font=("Segoe UI", 8)).pack(padx=8, pady=(6, 0))

        # === VLM Tab ===
        if self.vlm_controller:
            self.vlm_controller.build_vlm_tab(self.left_notebook)

        # === Dashboard Tab ===
        self.dashboard_ui = None
        try:
            from dashboard import DashboardUI
            self.dashboard_ui = DashboardUI(self)
            self.dashboard_ui.build_dashboard_tab(self.left_notebook)
        except ImportError:
            pass

        # === Train Tab ===
        self.train_ui = None
        try:
            from yolo_trainer import YOLOTrainerUI
            self.train_ui = YOLOTrainerUI(self)
            self.train_ui.build_train_tab(self.left_notebook)
        except ImportError:
            pass

        # Right Panel (Canvas)
        self.canvas_frame = tk.Frame(self.paned, bg="gray")
        self.paned.add(self.canvas_frame, minsize=400)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<Key>", self.on_key_press)

        # Click anywhere outside an Entry/Combobox → move focus to canvas
        # so keypresses don't accidentally type into text fields.
        # bind_all catches clicks on ALL widgets; we only steal focus when
        # the click target is not a text-input widget.
        def _steal_focus(event):
            w = event.widget
            # Don't steal focus from text inputs
            if isinstance(w, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
                return
            # Combobox dropdown popup: the internal listbox lives under a
            # path like ".!combobox.popdown.f.l" — check the widget path
            # string for "popdown" or "combobox" to catch all its children.
            try:
                wpath = str(w)
                if "popdown" in wpath or "combobox" in wpath.lower():
                    return
            except Exception:
                pass
            self.canvas.focus_set()
        self.root.bind_all("<Button-1>", _steal_focus, add="+")

    def _is_typing_in_entry(self):
        """Check if user is currently focused on a text input widget."""
        focused = self.root.focus_get()
        return isinstance(focused, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox))

    def on_key_press(self, event):
        # Don't intercept keys when user is typing in an input field
        if self._is_typing_in_entry():
            return
        if event.char.isdigit() and int(event.char) > 0:
            idx = int(event.char) - 1
            if idx < len(self.classes):
                self.listbox_classes.selection_clear(0, tk.END)
                self.listbox_classes.selection_set(idx)
                self.listbox_classes.event_generate("<<ListboxSelect>>")
        elif event.char.lower() == 'a':
            self.prev_image()
        elif event.char.lower() == 'd':
            self.next_image()

    def on_mouse_move(self, event):
        if not self.current_image: return
        self.canvas.delete("crosshair")
        
        self.crosshair_h = self.canvas.create_line(0, event.y, self.canvas.winfo_width(), event.y, fill="yellow", dash=(2, 2), tags="crosshair")
        self.crosshair_v = self.canvas.create_line(event.x, 0, event.x, self.canvas.winfo_height(), fill="yellow", dash=(2, 2), tags="crosshair")

    def update_class_listbox(self):
        self.listbox_classes.delete(0, tk.END)
        for i, c in enumerate(self.classes):
            self.listbox_classes.insert(tk.END, f"{i}: {c}")
        if self.classes:
            self.listbox_classes.selection_set(self.current_class_idx)
        # Keep VLM class dropdown in sync
        if self.vlm_controller:
            self.vlm_controller._refresh_class_dropdown()

    def load_images_dir(self):
        d = filedialog.askdirectory(title="Select Images Directory")
        if d:
            self.images_dir = d
            self.lbl_images_dir.config(text=f"Images: {os.path.basename(self.images_dir)}", fg="#333")
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in valid_exts]
            self.image_files.sort()
            self.current_index = 0
            
            if not self.labels_dir:
                self.labels_dir = d
                self.lbl_labels_dir.config(text=f"Labels: {os.path.basename(self.labels_dir)}", fg="#333")
            
            self.load_current_image()

    def load_labels_dir(self):
        d = filedialog.askdirectory(title="Select Labels Directory")
        if d:
            self.labels_dir = d
            self.lbl_labels_dir.config(text=f"Labels: {os.path.basename(self.labels_dir)}", fg="#333")
            if self.image_files:
                self.load_current_image()

    def load_video(self):
        """Open a video file, extract keyframes, and load them as images."""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                       ("All Files", "*.*")])
        if not video_path:
            return

        try:
            import cv2
        except ImportError:
            messagebox.showerror("Missing Dependency",
                "OpenCV is required for video support.\n\npip install opencv-python")
            return

        # Ask extraction method
        extract_win = tk.Toplevel(self.root)
        extract_win.title("Extract Keyframes")
        extract_win.geometry("400x280")
        extract_win.transient(self.root)
        extract_win.grab_set()

        FONT = ("Segoe UI", 10)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        tk.Label(extract_win, text=f"Video: {os.path.basename(video_path)}", font=("Segoe UI", 9, "bold")).pack(pady=(10, 2))
        tk.Label(extract_win, text=f"{total_frames} frames | {fps:.1f} fps | {duration:.1f}s", font=("Segoe UI", 9), fg="#555").pack()

        tk.Label(extract_win, text="Extraction method:", font=FONT).pack(pady=(10, 2))
        method_var = tk.StringVar(value="interval")
        methods_frame = tk.Frame(extract_win)
        methods_frame.pack(pady=2)
        ttk.Radiobutton(methods_frame, text="Every N frames", variable=method_var, value="interval").pack(anchor=tk.W)
        ttk.Radiobutton(methods_frame, text="Total N frames (uniform)", variable=method_var, value="count").pack(anchor=tk.W)

        param_frame = tk.Frame(extract_win)
        param_frame.pack(pady=6)
        tk.Label(param_frame, text="N:", font=FONT).pack(side=tk.LEFT)
        n_entry = tk.Entry(param_frame, width=8, font=FONT)
        n_entry.insert(0, str(max(1, int(fps))))  # default: 1 frame per second
        n_entry.pack(side=tk.LEFT, padx=6)

        # Output dir
        out_dir_var = tk.StringVar(value=os.path.join(os.path.dirname(video_path), f"{video_name}_frames"))
        dir_frame = tk.Frame(extract_win)
        dir_frame.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(dir_frame, text="Output:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(dir_frame, textvariable=out_dir_var, font=("Segoe UI", 8), fg="#555").pack(side=tk.LEFT, padx=4)

        result = {"go": False, "n_val": max(1, int(fps)), "method": "interval", "out_dir": ""}

        def on_extract():
            # Save values BEFORE destroying the window
            try:
                result["n_val"] = max(1, int(n_entry.get()))
            except ValueError:
                result["n_val"] = max(1, int(fps))
            result["method"] = method_var.get()
            result["out_dir"] = out_dir_var.get()
            result["go"] = True
            extract_win.destroy()

        tk.Button(extract_win, text="Extract & Load", font=FONT, bg="#a8d5a2",
                  relief=tk.GROOVE, command=on_extract).pack(pady=10)
        extract_win.wait_window()

        if not result["go"]:
            return

        n_val = result["n_val"]
        method = result["method"]
        out_dir = result["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        # Extract frames with progress
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Extracting Frames...")
        progress_win.geometry("400x100")
        progress_win.transient(self.root)
        lbl_prog = tk.Label(progress_win, text="Starting...")
        lbl_prog.pack(pady=10)
        pb = ttk.Progressbar(progress_win, mode="determinate", length=350)
        pb.pack(pady=10)
        self.root.update()

        def do_extract():
            cap = cv2.VideoCapture(video_path)
            extracted = []

            if method == "count":
                indices = [int(i * total_frames / n_val) for i in range(n_val)]
            else:
                indices = list(range(0, total_frames, n_val))

            pb.configure(maximum=len(indices))

            for count, frame_idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(out_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                extracted.append(frame_name)

                # Update progress directly (we're on the main thread)
                pb.configure(value=count + 1)
                lbl_prog.config(text=f"Extracting: {frame_name}")
                self.root.update_idletasks()

            cap.release()
            return extracted

        try:
            extracted_files = do_extract()
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("Extraction Error", str(e))
            return

        progress_win.destroy()

        if not extracted_files:
            messagebox.showinfo("No Frames", "No frames were extracted.")
            return

        # Load extracted frames as the image set
        self.images_dir = out_dir
        self.labels_dir = out_dir
        self.lbl_images_dir.config(text=f"Video: {video_name} ({len(extracted_files)} frames)", fg="#333")
        self.lbl_labels_dir.config(text=f"Labels: {os.path.basename(out_dir)}", fg="#333")
        self.image_files = sorted(extracted_files)
        self.current_index = 0
        self.load_current_image()

        messagebox.showinfo("Frames Extracted",
            f"Extracted {len(extracted_files)} frames from {os.path.basename(video_path)}\n"
            f"Saved to: {out_dir}")

    def load_classes_file(self):
        f = filedialog.askopenfilename(title="Select classes.txt", filetypes=[("Text Files", "*.txt")])
        if f:
            self.classes_file = f
            with open(f, 'r') as file:
                self.classes = [line.strip() for line in file if line.strip()]
            self.current_class_idx = 0
            self.update_class_listbox()

    def add_class(self):
        new_class = simpledialog.askstring("Add Class", "Enter new class name:")
        if new_class and new_class.strip():
            self.classes.append(new_class.strip())
            self.update_class_listbox()
            self.save_classes()

    def remove_class(self):
        sel = self.listbox_classes.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Select a class to remove.")
            return
        idx = sel[0]
        cls_name = self.classes[idx]
        if not messagebox.askyesno("Remove Class",
                f"Remove class '{cls_name}'?\n\n"
                "Existing annotations using this class will NOT be deleted,\n"
                "but their class index may become invalid."):
            return
        self.classes.pop(idx)
        if self.current_class_idx >= len(self.classes):
            self.current_class_idx = max(0, len(self.classes) - 1)
        self.update_class_listbox()
        self.save_classes()

    def save_classes(self):
        # Auto-save classes.txt to Labels dir if not explicitly loaded
        path = self.classes_file if self.classes_file else os.path.join(self.labels_dir, "classes.txt")
        if path and os.path.isdir(os.path.dirname(path)) if path else False:
            self.classes_file = path
            with open(path, 'w') as f:
                f.write("\n".join(self.classes))

    def on_class_select(self, event):
        sel = self.listbox_classes.curselection()
        if sel:
            self.current_class_idx = sel[0]
            
            # If an annotation is selected, update its class
            if self.selected_rect_idx is not None:
                self.annotations[self.selected_rect_idx]['class_idx'] = self.current_class_idx
                self.redraw_annotations()
                self.save_annotations()

    def on_annotation_select(self, event):
        sel = self.listbox_annotations.curselection()
        if sel:
            self.selected_rect_idx = sel[0]
        else:
            self.selected_rect_idx = None
        self.redraw_annotations()

    def delete_selected(self):
        if self.selected_rect_idx is not None:
            del self.annotations[self.selected_rect_idx]
            self.selected_rect_idx = None
            self.redraw_annotations()
            self.save_annotations()

    def discard_current_annotations(self):
        """Clear all annotations for the current image."""
        if not self.image_files:
            return
        self.annotations = []
        self.selected_rect_idx = None
        self.save_annotations()
        self.redraw_annotations()

    def discard_all_annotations(self):
        """Delete annotation .txt files for ALL loaded images."""
        if not self.image_files or not self.images_dir:
            return
        count = len(self.image_files)
        if not messagebox.askyesno(
            "Discard All",
            f"Delete annotation files for all {count} loaded images?\n\nThis cannot be undone."
        ):
            return
        removed = 0
        for rel in self.image_files:
            base = os.path.splitext(rel)[0]
            for d in {self.images_dir, getattr(self, "labels_dir", "") or ""}:
                if not d:
                    continue
                txt = os.path.join(d, base + ".txt")
                seg = os.path.join(d, base + "_seg.txt")
                for p in (txt, seg):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            removed += 1
                        except OSError:
                            pass
        self.annotations = []
        self.selected_rect_idx = None
        self.redraw_annotations()
        messagebox.showinfo("Done", f"Removed {removed} annotation file(s).")

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.save_annotations()
            if self.vlm_controller:
                self.vlm_controller.check_corrections_on_navigate()
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.image_files and self.current_index > 0:
            self.save_annotations()
            if self.vlm_controller:
                self.vlm_controller.check_corrections_on_navigate()
            self.current_index -= 1
            self.load_current_image()

    def jump_to_image(self, event=None):
        target = self.jump_entry.get().strip()
        if not target or not self.image_files:
            return
            
        # Try as index
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(self.image_files):
                self.save_annotations()
                self.current_index = idx
                self.load_current_image()
                return
                
        # Try as filename match
        for i, f in enumerate(self.image_files):
            if target.lower() in f.lower():
                self.save_annotations()
                self.current_index = i
                self.load_current_image()
                return
                
        messagebox.showinfo("Not Found", f"Could not find image matching: '{target}'")

    def export_annotations(self, up_to_current=False):
        if not self.images_dir or not self.labels_dir:
            messagebox.showwarning("Warning", "Please load images and labels first.")
            return
            
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            return
            
        import csv
        
        annotated_frames = 0
        annotated_videos = set()
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["Video", "Frame", "Total Annotations"]
                if self.classes:
                    header.extend(self.classes)
                writer.writerow(header)
                
                total_overall_boxes = 0
                total_class_counts = {c: 0 for c in range(len(self.classes))} if self.classes else {}
                
                limit = self.current_index + 1 if up_to_current else len(self.image_files)
                for i in range(limit):
                    img_name = self.image_files[i]
                    base_name = os.path.splitext(img_name)[0]
                    label_path = os.path.join(self.labels_dir, base_name + ".txt")
                    
                    lines = []
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as lf:
                            lines = lf.readlines()
                            
                    frame_name = os.path.basename(img_name)
                    if '_frame_' in frame_name:
                        video_name = frame_name.split('_frame_')[0]
                    elif '_frame' in frame_name:
                        video_name = frame_name.split('_frame')[0]
                    elif '_' in frame_name:
                        # Fallback: split by last underscore
                        video_name = frame_name.rsplit('_', 1)[0]
                    elif os.sep in img_name or '/' in img_name:
                        video_name = os.path.dirname(img_name)
                    else:
                        video_name = os.path.basename(os.path.normpath(self.images_dir))
                    
                    box_count = len(lines)
                    total_overall_boxes += box_count
                    
                    class_counts = {c: 0 for c in range(len(self.classes))}
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            try:
                                c_idx = int(parts[0])
                                if c_idx in class_counts:
                                    class_counts[c_idx] += 1
                                    total_class_counts[c_idx] += 1
                            except ValueError:
                                pass
                    
                    row = [video_name, frame_name, box_count]
                    if self.classes:
                        for c in range(len(self.classes)):
                            row.append(class_counts.get(c, 0))
                    
                    writer.writerow(row)
                    
                    annotated_frames += 1
                    annotated_videos.add(video_name)
                                
                # Append a summary row
                if annotated_frames > 0:
                    summary_row = ["TOTALS", f"{annotated_frames} Frames", total_overall_boxes]
                    if self.classes:
                        for c in range(len(self.classes)):
                            summary_row.append(total_class_counts.get(c, 0))
                    writer.writerow([])
                    writer.writerow(summary_row)
                                
            messagebox.showinfo("Export Successful", 
                f"Exported to {csv_path}\n\nTotal Videos Annotated: {len(annotated_videos)}\nTotal Frames Annotated: {annotated_frames}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def get_current_label_path(self):
        if not self.image_files or not self.labels_dir:
            return None
        img_name = self.image_files[self.current_index]
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.labels_dir, base_name + ".txt")

    def load_current_image(self):
        if not self.image_files:
            return
            
        self.annotations = []
        self.selected_rect_idx = None
        
        img_path = os.path.join(self.images_dir, self.image_files[self.current_index])
        try:
            self.current_image = Image.open(img_path)
            self.img_width, self.img_height = self.current_image.size
            self.lbl_progress.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
            
            # Load annotations
            label_path = self.get_current_label_path()
            if label_path and os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c_idx = int(parts[0])
                            # YOLO format: x_center, y_center, width, height (normalized)
                            xc, yc, w, h = map(float, parts[1:5])
                            
                            x_min = (xc - w/2) * self.img_width
                            y_min = (yc - h/2) * self.img_height
                            x_max = (xc + w/2) * self.img_width
                            y_max = (yc + h/2) * self.img_height
                            
                            self.annotations.append({'class_idx': c_idx, 'bbox': (x_min, y_min, x_max, y_max)})
                            
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error load_current_image", str(e))

    def display_image(self):
        if not self.current_image:
            return
            
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        if c_width <= 1 or c_height <= 1:
            # Not drawn yet, try updating
            self.root.update_idletasks()
            c_width = max(2, self.canvas.winfo_width())
            c_height = max(2, self.canvas.winfo_height())
            
        img_w, img_h = self.current_image.size
        
        self.scale_f = min(c_width / img_w, c_height / img_h)
        new_w = int(img_w * self.scale_f)
        new_h = int(img_h * self.scale_f)
        
        self.x_offset = (c_width - new_w) // 2
        self.y_offset = (c_height - new_h) // 2
        
        resized_img = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.tk_image)
        
        self.redraw_annotations()

    def on_window_resize(self, event):
        if event.widget == self.root and self.current_image:
            # Using after to debounce
            if hasattr(self, '_resize_job'):
                self.root.after_cancel(self._resize_job)
            self._resize_job = self.root.after(200, self.display_image)

    def redraw_annotations(self):
        self.canvas.delete("bbox")
        self.canvas.delete("handle")
        self.listbox_annotations.delete(0, tk.END)
        
        for i, ann in enumerate(self.annotations):
            cls_idx = ann['class_idx']
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # transform to canvas coordinates
            cx1 = x_min * self.scale_f + self.x_offset
            cy1 = y_min * self.scale_f + self.y_offset
            cx2 = x_max * self.scale_f + self.x_offset
            cy2 = y_max * self.scale_f + self.y_offset
            
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else f"Unknown ({cls_idx})"
            self.listbox_annotations.insert(tk.END, f"{cls_name} ({int(x_min)}, {int(y_min)})")
            
            base_color = COLORS[cls_idx % len(COLORS)]
            color = base_color if i != self.selected_rect_idx else "white"
            dash = () if i != self.selected_rect_idx else (4, 4)
            width = 5 if i == self.selected_rect_idx else 3
            
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width, tags="bbox", dash=dash)
            
            # Make label text more visible with a background rectangle
            text_id = self.canvas.create_text(cx1, cy1 - 10, text=cls_name, fill="white", anchor=tk.W, font=("Arial", 12, "bold"), tags="bbox")
            text_bbox = self.canvas.bbox(text_id)
            if text_bbox:
                bg_id = self.canvas.create_rectangle(text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2, fill=base_color, outline=base_color, tags="bbox")
                self.canvas.tag_lower(bg_id, text_id)
            
            # Draw resize handles if selected
            if i == self.selected_rect_idx:
                h_size = 4
                corners = [
                    (cx1, cy1), # TL
                    (cx2, cy1), # TR
                    (cx1, cy2), # BL
                    (cx2, cy2)  # BR
                ]
                for cx, cy in corners:
                    self.canvas.create_rectangle(cx-h_size, cy-h_size, cx+h_size, cy+h_size, fill="white", outline="black", tags="handle")
            
        if self.selected_rect_idx is not None:
            self.listbox_annotations.selection_set(self.selected_rect_idx)

    def _get_image_coords(self, canvas_x, canvas_y):
        img_x = (canvas_x - self.x_offset) / self.scale_f
        img_y = (canvas_y - self.y_offset) / self.scale_f
        # clamp
        img_x = max(0, min(img_x, self.img_width))
        img_y = max(0, min(img_y, self.img_height))
        return img_x, img_y

    def on_button_press(self, event):
        if not self.current_image: return
        
        img_x, img_y = self._get_image_coords(event.x, event.y)
        
        self.resize_handle_idx = None
        self.moving_rect = False
        
        # Check if clicked on a handle of the selected rect
        if self.selected_rect_idx is not None:
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # Handle hit detection
            h_size = 5
            corners = [
                (x_min, y_min), # TL 0
                (x_max, y_min), # TR 1
                (x_min, y_max), # BL 2
                (x_max, y_max)  # BR 3
            ]
            for idx, (cx, cy) in enumerate(corners):
                canvas_cx = cx * self.scale_f + self.x_offset
                canvas_cy = cy * self.scale_f + self.y_offset
                if abs(event.x - canvas_cx) <= h_size * 2 and abs(event.y - canvas_cy) <= h_size * 2:
                    self.resize_handle_idx = idx
                    self.start_x = img_x
                    self.start_y = img_y
                    return # start resizing
        
        # Check if Shift is held down (event.state & 0x0001) for forcing draw (optional now)
        force_draw = bool(event.state & 0x0001)
        
        # Check if clicked on outline of existing bbox
        clicked_idx = None
        if not force_draw:
            margin = 8 / self.scale_f  # 8 pixels tolerance for outline
            for i in range(len(self.annotations)-1, -1, -1):
                ann = self.annotations[i]
                x_min, y_min, x_max, y_max = ann['bbox']
                
                near_x = min(abs(img_x - x_min), abs(img_x - x_max)) <= margin
                near_y = min(abs(img_y - y_min), abs(img_y - y_max)) <= margin
                in_y_range = (y_min - margin) <= img_y <= (y_max + margin)
                in_x_range = (x_min - margin) <= img_x <= (x_max + margin)
                
                if (near_x and in_y_range) or (near_y and in_x_range):
                    clicked_idx = i
                    break
                
        if clicked_idx is not None:
            self.selected_rect_idx = clicked_idx
            self.moving_rect = True
            self.move_start_x = img_x
            self.move_start_y = img_y
            self.orig_bbox = self.annotations[clicked_idx]['bbox']
            self.redraw_annotations()
            return
            
        if not self.classes:
            messagebox.showwarning("Warning", "Please add or load classes first!")
            return
            
        # Start new drawing
        self.selected_rect_idx = None
        self.start_x = img_x
        self.start_y = img_y
        self.current_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=COLORS[self.current_class_idx % len(COLORS)], width=2, tags="bbox")

    def on_mouse_drag(self, event):
        if not self.current_image: return
        
        img_x, img_y = self._get_image_coords(event.x, event.y)
        self.on_mouse_move(event) # update crosshair
        
        if self.resize_handle_idx is not None and self.selected_rect_idx is not None:
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # modify bbox based on handle
            if self.resize_handle_idx == 0: x_min, y_min = img_x, img_y
            elif self.resize_handle_idx == 1: x_max, y_min = img_x, img_y
            elif self.resize_handle_idx == 2: x_min, y_max = img_x, img_y
            elif self.resize_handle_idx == 3: x_max, y_max = img_x, img_y
                
            # Avoid bounds flipping permanently during drag by ordering properly on release, but for rendering:
            render_xmin = min(x_min, x_max)
            render_xmax = max(x_min, x_max)
            render_ymin = min(y_min, y_max)
            render_ymax = max(y_min, y_max)
            
            self.annotations[self.selected_rect_idx]['bbox'] = (x_min, y_min, x_max, y_max)
            self.redraw_annotations()
            
        elif getattr(self, 'moving_rect', False) and self.selected_rect_idx is not None:
            # Move the entire bbox
            dx = img_x - self.move_start_x
            dy = img_y - self.move_start_y
            
            ox_min, oy_min, ox_max, oy_max = self.orig_bbox
            
            new_xmin = ox_min + dx
            new_ymin = oy_min + dy
            new_xmax = ox_max + dx
            new_ymax = oy_max + dy
            
            # Clamp to image boundaries
            if new_xmin < 0:
                new_xmax -= new_xmin
                new_xmin = 0
            if new_ymin < 0:
                new_ymax -= new_ymin
                new_ymin = 0
            if new_xmax > self.img_width:
                new_xmin -= (new_xmax - self.img_width)
                new_xmax = self.img_width
            if new_ymax > self.img_height:
                new_ymin -= (new_ymax - self.img_height)
                new_ymax = self.img_height
                
            self.annotations[self.selected_rect_idx]['bbox'] = (new_xmin, new_ymin, new_xmax, new_ymax)
            self.redraw_annotations()

        elif self.current_rect_id is not None:
            # Update canvas coords
            self.canvas.coords(self.current_rect_id, 
                               self.start_x * self.scale_f + self.x_offset, 
                               self.start_y * self.scale_f + self.y_offset, 
                               event.x, event.y)

    def on_button_release(self, event):
        if self.resize_handle_idx is not None and self.selected_rect_idx is not None:
            # Normalize bounding box coords after resize
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            self.annotations[self.selected_rect_idx]['bbox'] = (x_min, y_min, x_max, y_max)
            
            self.resize_handle_idx = None
            self.save_annotations()
            return
            
        if getattr(self, 'moving_rect', False):
            self.moving_rect = False
            self.save_annotations()
            return
            
        if self.current_rect_id is not None:
            end_x, end_y = self._get_image_coords(event.x, event.y)
            
            x_min = min(self.start_x, end_x)
            y_min = min(self.start_y, end_y)
            x_max = max(self.start_x, end_x)
            y_max = max(self.start_y, end_y)
            
            # Reject tiny boxes
            if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                self.annotations.append({
                    'class_idx': self.current_class_idx,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
                self.selected_rect_idx = len(self.annotations) - 1
                
            self.current_rect_id = None
            self.redraw_annotations()
            self.save_annotations()

    def save_annotations(self):
        label_path = self.get_current_label_path()
        if not label_path: return
        
        if not self.annotations:
            # Remove file if no annotations
            if os.path.exists(label_path):
                os.remove(label_path)
            return
            
        try:
            with open(label_path, 'w') as f:
                for ann in self.annotations:
                    x_min, y_min, x_max, y_max = ann['bbox']
                    
                    # YOLO conversion
                    w = (x_max - x_min) / self.img_width
                    h = (y_max - y_min) / self.img_height
                    xc = (x_min / self.img_width) + (w / 2)
                    yc = (y_min / self.img_height) + (h / 2)
                    
                    f.write(f"{ann['class_idx']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            messagebox.showerror("Error save_annotations", str(e))

    def auto_annotate(self):
        if not self.classes:
            messagebox.showwarning("Warning", "Please add or load classes first before auto-annotating, so we have something to map to!")
            return

        model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if not model_path:
            return

        # Ask: images directory or video file
        source_choice = tk.StringVar(value="images")
        src_win = tk.Toplevel(self.root)
        src_win.title("Select Source")
        src_win.geometry("320x140")
        src_win.transient(self.root)
        src_win.grab_set()
        tk.Label(src_win, text="Annotate from:", font=("Segoe UI", 10, "bold")).pack(pady=(12, 6))
        ttk.Radiobutton(src_win, text="Image directory", variable=source_choice, value="images").pack(anchor=tk.W, padx=40)
        ttk.Radiobutton(src_win, text="Video file (extract frames first)", variable=source_choice, value="video").pack(anchor=tk.W, padx=40)
        src_result = {"go": False}
        def _on_ok():
            src_result["go"] = True
            src_win.destroy()
        tk.Button(src_win, text="Continue", font=("Segoe UI", 10), bg="#a8d5a2",
                  relief=tk.GROOVE, command=_on_ok).pack(pady=10)
        src_win.wait_window()
        if not src_result["go"]:
            return

        # Collect image paths based on source type
        target_dir = None
        video_name = None
        if source_choice.get() == "video":
            target_dir = self._yolo_extract_video_frames()
            if not target_dir:
                return
            video_name = os.path.basename(target_dir).replace("_frames", "")
        else:
            target_dir = filedialog.askdirectory(title="Select Directory to Auto-Annotate")
            if not target_dir:
                return

        try:
            from ultralytics import YOLO
        except ImportError:
            messagebox.showerror("Import Error", "ultralytics package is required. Run 'pip install ultralytics'")
            return

        # Load model first to get its class names
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Loading...")
        progress_win.geometry("250x80")
        progress_win.transient(self.root)
        tk.Label(progress_win, text="Loading YOLO model names...").pack(pady=20)
        self.root.update()

        try:
            model = YOLO(model_path)
            model_names = model.names
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("Model Error", str(e))
            return

        progress_win.destroy()

        # Ask for mapping
        dialog = ClassMappingDialog(self.root, model_names, self.classes)
        self.root.wait_window(dialog)

        if not dialog.result:
            return

        class_mappings = dialog.result  # list of (model_idx, ws_idx)
        # Build a lookup: model_cls_id -> workspace_cls_id
        mapping_dict = {m: w for m, w in class_mappings}

        progress_win = tk.Toplevel(self.root)
        progress_win.title("Auto Annotating...")
        progress_win.geometry("400x120")
        progress_win.transient(self.root)
        progress_win.grab_set()

        lbl_info = tk.Label(progress_win, text="Loading model...")
        lbl_info.pack(pady=10)

        pb = ttk.Progressbar(progress_win, mode='determinate', length=300)
        pb.pack(pady=10)

        final_target_dir = target_dir
        final_video_name = video_name

        def run_inference():
            try:
                model = YOLO(model_path)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Model Error", str(e)))
                self.root.after(0, progress_win.destroy)
                return

            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            img_paths = []
            for root_dir, dirs, files in os.walk(final_target_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        img_paths.append(os.path.join(root_dir, f))

            total = len(img_paths)
            if total == 0:
                self.root.after(0, lambda: messagebox.showinfo("Done", "No images found."))
                self.root.after(0, progress_win.destroy)
                return

            pb.configure(maximum=total)
            count = 0

            for img_path in img_paths:
                try:
                    self.root.after(0, lambda p=img_path: lbl_info.config(text=f"Processing: {os.path.basename(p)}"))
                    results = model(img_path, verbose=False)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"

                    # Read existing annotations if any to keep them, we will append new ones
                    existing_lines = []
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as ext_f:
                            existing_lines = ext_f.readlines()

                    new_boxes = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            if cls_id in mapping_dict:
                                ws_idx = mapping_dict[cls_id]
                                x, y, w, h = box.xywhn[0].tolist()
                                new_boxes.append(f"{ws_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                    if existing_lines or new_boxes:
                        with open(txt_path, 'w') as out_f:
                            for ln in existing_lines:
                                out_f.write(ln)
                            for nb in new_boxes:
                                out_f.write(nb)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

                count += 1
                self.root.after(0, lambda c=count: pb.configure(value=c))

            def on_success():
                src_label = f"Video: {final_video_name}" if final_video_name else f"Images: {os.path.basename(final_target_dir)}"
                map_summary = ", ".join(
                    f"{model_names.get(m, m)} → {self.classes[w]}" for m, w in class_mappings
                )
                messagebox.showinfo("Success",
                    f"Auto-annotated {count} images!\n\nMappings:\n{map_summary}")
                # Automatically open the target dir in the tool
                self.images_dir = final_target_dir
                self.labels_dir = final_target_dir
                self.lbl_images_dir.config(text=f"{src_label} ({count} frames)", fg="#333")
                self.lbl_labels_dir.config(text=f"Labels: {os.path.basename(final_target_dir)}", fg="#333")

                # Load all images recursively with relative paths
                valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                all_rel_paths = []
                for root_dir, _, files in os.walk(final_target_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            abs_path = os.path.join(root_dir, f)
                            rel_path = os.path.relpath(abs_path, final_target_dir)
                            all_rel_paths.append(rel_path)

                self.image_files = sorted(all_rel_paths)
                self.current_index = 0

                if self.image_files:
                    self.load_current_image()

                progress_win.destroy()

            self.root.after(0, on_success)

        threading.Thread(target=run_inference, daemon=True).start()

    def _yolo_extract_video_frames(self):
        """Extract frames from a video for YOLO auto-annotation. Returns output dir or None."""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
                       ("All Files", "*.*")])
        if not video_path:
            return None

        try:
            import cv2
        except ImportError:
            messagebox.showerror("Missing Dependency",
                "OpenCV is required for video support.\n\npip install opencv-python")
            return None

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Extraction settings dialog
        extract_win = tk.Toplevel(self.root)
        extract_win.title("Extract Keyframes")
        extract_win.geometry("400x280")
        extract_win.transient(self.root)
        extract_win.grab_set()
        FONT = ("Segoe UI", 10)

        tk.Label(extract_win, text=f"Video: {os.path.basename(video_path)}",
                 font=("Segoe UI", 9, "bold")).pack(pady=(10, 2))
        tk.Label(extract_win, text=f"{total_frames} frames | {fps:.1f} fps | {duration:.1f}s",
                 font=("Segoe UI", 9), fg="#555").pack()

        tk.Label(extract_win, text="Extraction method:", font=FONT).pack(pady=(10, 2))
        method_var = tk.StringVar(value="interval")
        mf = tk.Frame(extract_win)
        mf.pack(pady=2)
        ttk.Radiobutton(mf, text="Every N frames", variable=method_var, value="interval").pack(anchor=tk.W)
        ttk.Radiobutton(mf, text="Total N frames (uniform)", variable=method_var, value="count").pack(anchor=tk.W)

        pf = tk.Frame(extract_win)
        pf.pack(pady=6)
        tk.Label(pf, text="N:", font=FONT).pack(side=tk.LEFT)
        n_entry = tk.Entry(pf, width=8, font=FONT)
        n_entry.insert(0, str(max(1, int(fps))))
        n_entry.pack(side=tk.LEFT, padx=6)

        out_dir_var = tk.StringVar(value=os.path.join(os.path.dirname(video_path), f"{video_name}_frames"))
        df = tk.Frame(extract_win)
        df.pack(fill=tk.X, padx=20, pady=4)
        tk.Label(df, text="Output:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(df, textvariable=out_dir_var, font=("Segoe UI", 8), fg="#555").pack(side=tk.LEFT, padx=4)

        result = {"go": False, "n_val": max(1, int(fps)), "method": "interval", "out_dir": ""}
        def on_extract():
            try:
                result["n_val"] = max(1, int(n_entry.get()))
            except ValueError:
                result["n_val"] = max(1, int(fps))
            result["method"] = method_var.get()
            result["out_dir"] = out_dir_var.get()
            result["go"] = True
            extract_win.destroy()

        tk.Button(extract_win, text="Extract", font=FONT, bg="#a8d5a2",
                  relief=tk.GROOVE, command=on_extract).pack(pady=10)
        extract_win.wait_window()

        if not result["go"]:
            return None

        n_val = result["n_val"]
        method = result["method"]
        out_dir = result["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        # Extract frames with progress
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Extracting Frames...")
        progress_win.geometry("400x100")
        progress_win.transient(self.root)
        lbl_prog = tk.Label(progress_win, text="Starting...")
        lbl_prog.pack(pady=10)
        pb = ttk.Progressbar(progress_win, mode="determinate", length=350)
        pb.pack(pady=10)
        self.root.update()

        cap = cv2.VideoCapture(video_path)
        if method == "count":
            indices = [int(i * total_frames / n_val) for i in range(n_val)]
        else:
            indices = list(range(0, total_frames, n_val))

        pb.configure(maximum=len(indices))
        extracted = 0

        for count, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, frame_name), frame)
            extracted += 1
            pb.configure(value=count + 1)
            lbl_prog.config(text=f"Extracting: {frame_name}")
            self.root.update_idletasks()

        cap.release()
        progress_win.destroy()

        if extracted == 0:
            messagebox.showinfo("No Frames", "No frames were extracted.")
            return None

        messagebox.showinfo("Frames Extracted",
            f"Extracted {extracted} frames from {os.path.basename(video_path)}\nSaved to: {out_dir}")
        return out_dir

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotationTool(root)
    root.mainloop()
