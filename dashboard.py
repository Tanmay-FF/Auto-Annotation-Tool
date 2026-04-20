"""
Annotation Quality Dashboard for the annotation tool.

Provides visual analytics:
  - Class distribution bar chart
  - Annotation coverage metrics
  - Per-image box count histogram
  - Empty/sparse image detection
  - Summary statistics cards

Uses matplotlib for charts (graceful fallback if not installed).

Usage:
    from dashboard import DashboardUI
    dashboard = DashboardUI(tool)
    dashboard.build_dashboard_tab(notebook)
"""

import os
import tkinter as tk
from tkinter import ttk

# ── Theme constants (Light) ──────────────────────────────────────────────────

BG = "#f0f2f5"
BG_CARD = "#ffffff"
BG_CARD_ALT = "#f7f8fa"
FG = "#1e293b"
FG_DIM = "#94a3b8"
FG_ACCENT = "#3b82f6"
FG_GREEN = "#16a34a"
FG_PEACH = "#ea580c"
FG_RED = "#dc2626"
FG_YELLOW = "#ca8a04"
FG_MAUVE = "#7c3aed"
FONT = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_LG = ("Segoe UI", 11, "bold")
FONT_XL = ("Segoe UI", 18, "bold")
FONT_SM = ("Segoe UI", 8)

# Chart colors (modern light palette)
CHART_COLORS = [
    "#3b82f6", "#16a34a", "#ea580c", "#dc2626", "#7c3aed",
    "#ca8a04", "#0891b2", "#2563eb", "#db2777", "#0d9488",
    "#6366f1", "#e11d48", "#64748b", "#f59e0b", "#475569",
]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ── Statistics Scanner ────────────────────────────────────────────────────────

class AnnotationStats:
    """Scans YOLO label files and computes annotation statistics."""

    def __init__(self, images_dir, labels_dir, image_files, classes):
        self.images_dir = images_dir
        self.labels_dir = labels_dir or images_dir
        self.image_files = image_files or []
        self.classes = classes or []

    def _find_label_path(self, rel):
        """Find the label .txt file for an image, using the same logic as
        get_current_label_path() and load_current_image() in the main tool.

        Priority:
          1. labels_dir + relative path (preserving subdirs) — matches get_current_label_path()
          2. Same directory as the image — matches auto-annotate output
        """
        base_rel = os.path.splitext(rel)[0] + ".txt"

        # Primary: labels_dir (same path structure as images)
        if self.labels_dir:
            p = os.path.join(self.labels_dir, base_rel)
            if os.path.exists(p):
                return p

        # Fallback: next to the image file
        p = os.path.join(self.images_dir, base_rel)
        if os.path.exists(p):
            return p

        return None

    def scan(self):
        class_counts = {}
        per_image = []  # (filename, box_count)
        empty_images = []
        total_boxes = 0
        annotated = 0
        class_per_image = {}  # class_idx -> set of image indices

        for idx, rel in enumerate(self.image_files):
            txt = self._find_label_path(rel)

            box_count = 0
            if txt:
                try:
                    with open(txt, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                cls_idx = int(parts[0])
                                # Validate the remaining 4 values are valid floats
                                float(parts[1]); float(parts[2])
                                float(parts[3]); float(parts[4])
                            except (ValueError, IndexError):
                                continue
                            class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
                            if cls_idx not in class_per_image:
                                class_per_image[cls_idx] = set()
                            class_per_image[cls_idx].add(idx)
                            box_count += 1
                except OSError:
                    pass

            per_image.append((rel, box_count))
            total_boxes += box_count
            if box_count > 0:
                annotated += 1
            else:
                empty_images.append(rel)

        total = len(self.image_files)
        box_counts = [c for _, c in per_image]

        # Verify totals are consistent
        assert total_boxes == sum(class_counts.values()), \
            f"Mismatch: total_boxes={total_boxes}, sum(class_counts)={sum(class_counts.values())}"

        return {
            "total_images": total,
            "annotated_images": annotated,
            "unannotated_images": total - annotated,
            "total_boxes": total_boxes,
            "avg_boxes": total_boxes / max(1, annotated),
            "class_counts": class_counts,
            "class_per_image": class_per_image,
            "per_image": per_image,
            "box_counts": box_counts,
            "empty_images": empty_images,
            "max_boxes": max(box_counts) if box_counts else 0,
            "min_boxes_annotated": min((c for c in box_counts if c > 0), default=0),
        }


# ── Dashboard UI ─────────────────────────────────────────────────────────────

class DashboardUI:
    """Builds the Dashboard tab with modern dark-themed analytics."""

    def __init__(self, tool):
        self.tool = tool
        self.root = tool.root
        self._stats = None
        self._chart_canvases = []  # matplotlib canvases to destroy on refresh

    def build_dashboard_tab(self, notebook: ttk.Notebook):
        outer = tk.Frame(notebook, bg=BG)
        notebook.add(outer, text="  Dashboard  ")

        # Scrollable canvas
        _canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        _vsb = ttk.Scrollbar(outer, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.pack(side=tk.RIGHT, fill=tk.Y)
        _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._content = tk.Frame(_canvas, bg=BG)
        _win = _canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._content.bind("<Configure>",
            lambda e: _canvas.configure(scrollregion=_canvas.bbox("all")))
        _canvas.bind("<Configure>",
            lambda e: _canvas.itemconfig(_win, width=e.width))

        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        _canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        self._canvas_widget = _canvas

        # Header
        hdr = tk.Frame(self._content, bg=BG)
        hdr.pack(fill=tk.X, padx=12, pady=(12, 4))
        tk.Label(hdr, text="Annotation Dashboard", bg=BG, fg=FG,
                 font=FONT_LG).pack(side=tk.LEFT)
        tk.Button(hdr, text="Refresh", bg=FG_ACCENT, fg="#ffffff",
                  font=FONT_BOLD, relief=tk.FLAT, padx=12, pady=2,
                  cursor="hand2", command=self._refresh).pack(side=tk.RIGHT)

        # Container for dynamic content
        self._cards_frame = tk.Frame(self._content, bg=BG)
        self._cards_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Initial empty state
        self._show_empty_state()

        return outer

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self):
        images_dir = getattr(self.tool, "images_dir", "") or ""
        classes = getattr(self.tool, "classes", [])
        image_files = getattr(self.tool, "image_files", [])
        labels_dir = getattr(self.tool, "labels_dir", "") or ""

        if not images_dir or not image_files:
            self._show_empty_state()
            return

        scanner = AnnotationStats(images_dir, labels_dir, image_files, classes)
        self._stats = scanner.scan()
        self._render(self._stats, classes)

    def _clear_cards(self):
        for c in self._chart_canvases:
            try:
                c.get_tk_widget().destroy()
            except Exception:
                pass
        self._chart_canvases.clear()
        for w in self._cards_frame.winfo_children():
            w.destroy()

    def _show_empty_state(self):
        self._clear_cards()
        msg = tk.Frame(self._cards_frame, bg=BG_CARD, padx=20, pady=30)
        msg.pack(fill=tk.X, padx=8, pady=20)
        tk.Label(msg, text="No Data Loaded", bg=BG_CARD, fg=FG_DIM,
                 font=FONT_LG).pack()
        tk.Label(msg, text="Load images and click Refresh to see analytics.",
                 bg=BG_CARD, fg=FG_DIM, font=FONT).pack(pady=(4, 0))

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self, stats, classes):
        self._clear_cards()

        # Row 1: Summary cards
        self._render_summary_cards(stats)

        # Row 2: Class distribution chart
        self._render_class_distribution(stats, classes)

        # Row 3: Box count histogram + Coverage donut side by side
        row3 = tk.Frame(self._cards_frame, bg=BG)
        row3.pack(fill=tk.X, padx=4, pady=4)
        self._render_box_histogram(stats, row3)
        self._render_coverage_donut(stats, row3)

        # Row 4: Per-class image coverage
        self._render_class_image_coverage(stats, classes)

        # Row 5: Empty / sparse images
        self._render_flagged_images(stats)

    # ── Summary Cards ─────────────────────────────────────────────────────────

    def _render_summary_cards(self, stats):
        row = tk.Frame(self._cards_frame, bg=BG)
        row.pack(fill=tk.X, padx=4, pady=(4, 8))

        cards = [
            ("Total Images", str(stats["total_images"]), FG_ACCENT),
            ("Annotated", str(stats["annotated_images"]), FG_GREEN),
            ("Unannotated", str(stats["unannotated_images"]),
             FG_RED if stats["unannotated_images"] > 0 else FG_DIM),
            ("Total Boxes", str(stats["total_boxes"]), FG_YELLOW),
            ("Avg / Image", f"{stats['avg_boxes']:.1f}", FG_MAUVE),
            ("Classes", str(len(stats["class_counts"])), FG_PEACH),
        ]

        for title, value, color in cards:
            card = tk.Frame(row, bg=BG_CARD, padx=10, pady=8)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            tk.Label(card, text=value, bg=BG_CARD, fg=color,
                     font=FONT_XL).pack()
            tk.Label(card, text=title, bg=BG_CARD, fg=FG_DIM,
                     font=FONT_SM).pack()

    # ── Class Distribution Bar Chart ──────────────────────────────────────────

    def _render_class_distribution(self, stats, classes):
        card = tk.Frame(self._cards_frame, bg=BG_CARD, padx=12, pady=10)
        card.pack(fill=tk.X, padx=4, pady=4)

        hdr = tk.Frame(card, bg=BG_CARD)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Class Distribution", bg=BG_CARD, fg=FG,
                 font=FONT_BOLD).pack(side=tk.LEFT)

        cc = stats["class_counts"]
        if not cc:
            tk.Label(card, text="No annotations yet", bg=BG_CARD, fg=FG_DIM,
                     font=FONT).pack(pady=8)
            return

        # Verify: sum of class counts should equal total boxes
        total_from_classes = sum(cc.values())
        tk.Label(hdr, text=f"Total: {total_from_classes} boxes across {len(cc)} classes",
                 bg=BG_CARD, fg=FG_DIM, font=FONT_SM).pack(side=tk.RIGHT)

        # Sort by count descending — assign stable color per class index
        sorted_cls = sorted(cc.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_cls[0][1] if sorted_cls else 1

        if _MPL_AVAILABLE:
            self._render_class_chart_mpl(card, sorted_cls, classes, max_count)
        else:
            self._render_class_chart_text(card, sorted_cls, classes, max_count)

    def _render_class_chart_mpl(self, parent, sorted_cls, classes, max_count):
        """sorted_cls is [(cls_idx, count), ...] sorted descending by count.
        We reverse for barh so the largest bar appears at the top."""
        n = len(sorted_cls)
        fig = Figure(figsize=(3.2, max(1.2, 0.35 * n)), dpi=100, facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)

        # Build data in bottom-to-top order (reversed) for barh display
        # but assign colors based on the original sorted order so
        # color[0] = most frequent class
        display_order = list(reversed(sorted_cls))
        names = []
        counts = []
        colors = []
        for cls_idx, count in display_order:
            name = classes[cls_idx] if cls_idx < len(classes) else f"Unknown({cls_idx})"
            names.append(f"{name} ({count})")
            counts.append(count)
            # Color by original rank: find position in sorted_cls
            rank = next(i for i, (c, _) in enumerate(sorted_cls) if c == cls_idx)
            colors.append(CHART_COLORS[rank % len(CHART_COLORS)])

        bars = ax.barh(range(n), counts, color=colors, height=0.6, edgecolor="none")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8, color=FG)
        ax.set_xlim(0, max_count * 1.15)
        ax.tick_params(axis='x', colors=FG_DIM, labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(FG_DIM)
        ax.spines['left'].set_visible(False)

        # Percentage labels on bars
        total_boxes = sum(counts)
        for bar, count in zip(bars, counts):
            pct = count / max(1, total_boxes) * 100
            ax.text(bar.get_width() + max_count * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va='center', fontsize=7, color=FG_DIM)

        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X, pady=(4, 0))
        self._chart_canvases.append(canvas)

    def _render_class_chart_text(self, parent, sorted_cls, classes, max_count):
        """Fallback text-based bar chart when matplotlib is not available."""
        BAR_WIDTH = 20
        for i, (cls_idx, count) in enumerate(sorted_cls):
            name = classes[cls_idx] if cls_idx < len(classes) else f"Unknown({cls_idx})"
            pct = count / max(1, max_count)
            filled = int(pct * BAR_WIDTH)
            bar_str = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)
            color = CHART_COLORS[i % len(CHART_COLORS)]

            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{name:>15s}", bg=BG_CARD, fg=FG,
                     font=("Consolas", 8), width=15, anchor=tk.E).pack(side=tk.LEFT)
            tk.Label(row, text=f" {bar_str} ", bg=BG_CARD, fg=color,
                     font=("Consolas", 8)).pack(side=tk.LEFT)
            tk.Label(row, text=str(count), bg=BG_CARD, fg=FG,
                     font=("Consolas", 8, "bold")).pack(side=tk.LEFT)

    # ── Box Count Histogram ───────────────────────────────────────────────────

    def _render_box_histogram(self, stats, parent):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=10)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        tk.Label(card, text="Boxes Per Image", bg=BG_CARD, fg=FG,
                 font=FONT_BOLD).pack(anchor=tk.W)

        box_counts = stats["box_counts"]
        if not box_counts or max(box_counts) == 0:
            tk.Label(card, text="No data", bg=BG_CARD, fg=FG_DIM,
                     font=FONT).pack(pady=8)
            return

        if _MPL_AVAILABLE:
            fig = Figure(figsize=(1.8, 1.4), dpi=100, facecolor=BG_CARD)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)

            max_val = max(box_counts)
            bins = min(max_val + 1, 20)
            ax.hist(box_counts, bins=bins, color=FG_ACCENT, edgecolor=BG_CARD,
                    alpha=0.85, linewidth=0.5)
            ax.set_xlabel("boxes", fontsize=7, color=FG_DIM)
            ax.set_ylabel("images", fontsize=7, color=FG_DIM)
            ax.tick_params(axis='both', colors=FG_DIM, labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(FG_DIM)
            ax.spines['left'].set_color(FG_DIM)
            fig.tight_layout(pad=0.5)

            canvas = FigureCanvasTkAgg(fig, master=card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))
            self._chart_canvases.append(canvas)
        else:
            # Text fallback: bucket summary
            buckets = {"0": 0, "1-3": 0, "4-6": 0, "7-10": 0, "11+": 0}
            for c in box_counts:
                if c == 0:
                    buckets["0"] += 1
                elif c <= 3:
                    buckets["1-3"] += 1
                elif c <= 6:
                    buckets["4-6"] += 1
                elif c <= 10:
                    buckets["7-10"] += 1
                else:
                    buckets["11+"] += 1
            for label, cnt in buckets.items():
                tk.Label(card, text=f"{label} boxes: {cnt} images", bg=BG_CARD,
                         fg=FG, font=FONT_SM).pack(anchor=tk.W)

    # ── Coverage Donut ────────────────────────────────────────────────────────

    def _render_coverage_donut(self, stats, parent):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=10)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        tk.Label(card, text="Annotation Coverage", bg=BG_CARD, fg=FG,
                 font=FONT_BOLD).pack(anchor=tk.W)

        ann = stats["annotated_images"]
        unann = stats["unannotated_images"]
        total = stats["total_images"]
        pct = (ann / max(1, total)) * 100

        if _MPL_AVAILABLE:
            fig = Figure(figsize=(1.8, 1.4), dpi=100, facecolor=BG_CARD)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_CARD)

            sizes = [ann, unann] if unann > 0 else [ann]
            colors_d = [FG_GREEN, "#45475a"] if unann > 0 else [FG_GREEN]
            wedges, _ = ax.pie(sizes, colors=colors_d, startangle=90,
                                wedgeprops=dict(width=0.35, edgecolor=BG_CARD))
            ax.text(0, 0, f"{pct:.0f}%", ha='center', va='center',
                    fontsize=14, fontweight='bold', color=FG)
            fig.tight_layout(pad=0.3)

            canvas = FigureCanvasTkAgg(fig, master=card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(4, 0))
            self._chart_canvases.append(canvas)
        else:
            tk.Label(card, text=f"{pct:.0f}%", bg=BG_CARD, fg=FG_GREEN,
                     font=FONT_XL).pack(pady=(8, 2))

        tk.Label(card, text=f"{ann} / {total} images annotated", bg=BG_CARD,
                 fg=FG_DIM, font=FONT_SM).pack()

    # ── Per-Class Image Coverage ──────────────────────────────────────────────

    def _render_class_image_coverage(self, stats, classes):
        cpi = stats.get("class_per_image", {})
        if not cpi:
            return

        card = tk.Frame(self._cards_frame, bg=BG_CARD, padx=12, pady=10)
        card.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(card, text="Class Presence Across Images", bg=BG_CARD, fg=FG,
                 font=FONT_BOLD).pack(anchor=tk.W)
        tk.Label(card, text="How many images contain each class",
                 bg=BG_CARD, fg=FG_DIM, font=FONT_SM).pack(anchor=tk.W, pady=(0, 6))

        total = stats["total_images"]
        sorted_cls = sorted(cpi.items(), key=lambda x: len(x[1]), reverse=True)

        for i, (cls_idx, img_set) in enumerate(sorted_cls):
            name = classes[cls_idx] if cls_idx < len(classes) else f"Unknown({cls_idx})"
            count = len(img_set)
            pct = count / max(1, total) * 100
            color = CHART_COLORS[i % len(CHART_COLORS)]

            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=name, bg=BG_CARD, fg=FG, font=FONT,
                     width=14, anchor=tk.W).pack(side=tk.LEFT)

            # Progress bar background
            bar_bg = tk.Frame(row, bg="#e2e8f0", height=14)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
            bar_bg.pack_propagate(False)

            # Filled portion
            fill_pct = max(0.01, pct / 100.0)
            bar_fill = tk.Frame(bar_bg, bg=color)
            bar_fill.place(relx=0, rely=0, relwidth=fill_pct, relheight=1.0)

            tk.Label(row, text=f"{count} ({pct:.0f}%)", bg=BG_CARD, fg=FG_DIM,
                     font=FONT_SM, width=10).pack(side=tk.RIGHT)

    # ── Flagged Images ────────────────────────────────────────────────────────

    def _render_flagged_images(self, stats):
        empty = stats["empty_images"]
        per_image = stats["per_image"]

        # Find sparse images (annotated but very few boxes compared to average)
        avg = stats["avg_boxes"]
        sparse = []
        if avg > 2:
            sparse = [(f, c) for f, c in per_image if 0 < c < avg * 0.3]
            sparse.sort(key=lambda x: x[1])

        # Find dense outliers
        dense = []
        if avg > 0:
            dense = [(f, c) for f, c in per_image if c > avg * 3 and c > 5]
            dense.sort(key=lambda x: x[1], reverse=True)

        if not empty and not sparse and not dense:
            return

        card = tk.Frame(self._cards_frame, bg=BG_CARD, padx=12, pady=10)
        card.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(card, text="Flagged Images", bg=BG_CARD, fg=FG,
                 font=FONT_BOLD).pack(anchor=tk.W)
        tk.Label(card, text="Images that may need attention (click to navigate)",
                 bg=BG_CARD, fg=FG_DIM, font=FONT_SM).pack(anchor=tk.W, pady=(0, 6))

        if empty:
            self._render_flag_section(card, f"No Annotations ({len(empty)})",
                                       FG_RED, empty[:20], show_count=False)
        if sparse:
            self._render_flag_section(card, f"Sparse (< {avg*0.3:.0f} boxes, {len(sparse)} images)",
                                       FG_YELLOW,
                                       [(f, f"  {c} boxes") for f, c in sparse[:15]])
        if dense:
            self._render_flag_section(card, f"Dense Outliers (> {avg*3:.0f} boxes, {len(dense)} images)",
                                       FG_PEACH,
                                       [(f, f"  {c} boxes") for f, c in dense[:15]])

    def _render_flag_section(self, parent, title, color, items, show_count=True):
        tk.Label(parent, text=f"\u2022 {title}", bg=BG_CARD, fg=color,
                 font=FONT_BOLD).pack(anchor=tk.W, pady=(4, 2))

        listbox = tk.Listbox(parent, height=min(6, len(items)),
                              bg=BG_CARD_ALT, fg=FG, font=("Consolas", 8),
                              selectbackground=FG_ACCENT, selectforeground="#ffffff",
                              relief=tk.FLAT, bd=0, highlightthickness=0)
        listbox.pack(fill=tk.X, pady=(0, 4))

        file_list = []
        for item in items:
            if isinstance(item, tuple):
                fname, suffix = item
                listbox.insert(tk.END, f"  {os.path.basename(fname)}{suffix}")
            else:
                fname = item
                listbox.insert(tk.END, f"  {os.path.basename(fname)}")
            file_list.append(fname)

        def on_click(event):
            sel = listbox.curselection()
            if not sel:
                return
            fname = file_list[sel[0]]
            # Navigate to this image
            image_files = getattr(self.tool, "image_files", [])
            for i, f in enumerate(image_files):
                if f == fname or os.path.basename(f) == os.path.basename(fname):
                    self.tool.current_index = i
                    self.tool.load_current_image()
                    # Switch to Annotate tab
                    try:
                        self.tool.left_notebook.select(0)
                    except Exception:
                        pass
                    break

        listbox.bind("<<ListboxSelect>>", on_click)
