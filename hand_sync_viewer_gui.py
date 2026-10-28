import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, traceback
import multiprocessing
import gc
import threading
import queue
from mediapipe_extraction import extraction_worker, draw_overlay

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

try:
    from correct_tap_detection import TapExtractor
    from feature_extraction import FeatureExtractor, extract_hand_features
    _PIPELINE_OK  = True
    _PIPELINE_ERR = ""
except ImportError as _ie:
    _PIPELINE_OK  = False
    _PIPELINE_ERR = str(_ie)

_BG     = "#0e0e1a"
_PANEL  = "#13131f"
_AX_BG  = "#191928"
_BORDER = "#2a2a45"
_TEXT   = "#c8c8e0"
_MUTED  = "#555577"
_ACCENT = "#4fc3f7"

_COL_AMP  = "#4fc3f7"
_COL_OPEN = "#42a5f5"
_COL_CLOS = "#ef5350"
_COL_SPD  = "#ab47bc"
_COL_TAP  = "#ffd54f"
_COL_CUR  = "#ff5252"
_COL_GRID = "#1e1e32"
_OK_GREEN = "#81c784"
_ERR_RED  = "#ef9a9a"


def _fmt(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s//60):02d}:{s%60:05.2f}"


def _display_scale():
    return 1.0


def _sym_limit(x, pad=1.1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    m = float(np.max(np.abs(x)))
    return m * pad if m > 0 else 1.0


class HandSyncViewer:
    VID_W = 640
    VID_H = 480

    def __init__(self, root: tk.Tk):
        # Tk objects must never be finalized by an arbitrary worker's cyclic GC.
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("HandSyncViewer must be created on the main thread")
        gc.disable()
        self._closing = False
        self.root = root
        self.root.report_callback_exception = self._callback_error
        self.root.title("Hand Sync Viewer")
        self.root.configure(bg=_BG)
        self.root.minsize(1180, 640)

        self._video_path = ""
        self._overlay = []
        self._extracting = False
        self._mp_context = multiprocessing.get_context("spawn")
        self._cancel_extract = self._mp_context.Event()
        self._extract_queue = self._mp_context.Queue()
        self._extract_process = None
        self.cap            = None
        self.src_fps        = 30.0
        self.src_frames     = 0
        self.src_w          = self.VID_W
        self.src_h          = self.VID_H
        self.current_fidx   = 0
        self.playing        = False
        self.play_speed     = 1.0
        self._after_id      = None
        self._last_raw      = None
        self._photo         = None
        self._placeholder_id= None
        self._suppress_seek = False

        self.zoom_rect   = None
        self._drag_start = None
        self._rect_id    = None

        self._activity = None
        self._hand_data = {"Right": None, "Left": None}

        self.sig_t   = None
        self.sig_amp = None
        self.sig_vel = None
        self.sig_spd = None
        self.sig_tap = None
        self._t_offset = 0.0

        self._fig         = None
        self._axes        = []
        self._cursors     = []
        self._wave_canvas = None

        self._xlsx_path = ""

        self._build_ui()
        self._build_wave_figure()
        self._draw_placeholders()

        self.root.bind("<space>",  lambda e: self._toggle_play())
        self.root.bind("<Right>",  lambda e: self._step(+1))
        self.root.bind("<Left>",   lambda e: self._step(-1))
        self.root.bind("r",        lambda e: self._reset_zoom())
        self.root.bind("R",        lambda e: self._reset_zoom())
        self.root.bind("q",        lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(1000, self._collect_ui_cycles)

        if not _PIPELINE_OK:
            messagebox.showwarning(
                "Pipeline modules not found",
                f"Could not import:\n{_PIPELINE_ERR}\n\n"
                "Place correct_tap_detection.py and feature_extraction.py next to this script."
            )

    def _build_ui(self):

        def _btn(p, text, cmd, w=None):
            kw = {"width": w} if w else {}
            b = tk.Button(p, text=text, command=cmd,
                          bg=_BORDER, fg=_TEXT, relief=tk.FLAT,
                          activebackground="#3a3a60", activeforeground="white",
                          padx=9, pady=4, font=("Segoe UI", 9),
                          cursor="hand2", **kw)
            b.pack(side=tk.LEFT, padx=2)
            return b

        def _sep(p):
            tk.Frame(p, bg="#2a2a50", width=1, height=24
                     ).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        def _lbl(p, t, fg=None):
            tk.Label(p, text=t, bg=p["bg"],
                     fg=fg or _MUTED, font=("Segoe UI", 9)
                     ).pack(side=tk.LEFT, padx=(0, 2))

        def _ent(p, var, w=6):
            e = tk.Entry(p, textvariable=var, width=w,
                         bg=_BORDER, fg=_TEXT, insertbackground=_TEXT,
                         relief=tk.FLAT, font=("Consolas", 9))
            e.pack(side=tk.LEFT, padx=(0, 3))
            return e

        r1 = tk.Frame(self.root, bg=_PANEL, pady=5, padx=8)
        r1.pack(side=tk.TOP, fill=tk.X)

        _btn(r1, "📂  Open Video", self._open_video)
        _sep(r1)
        self._play_btn = _btn(r1, "▶  Play", self._toggle_play, w=10)
        _btn(r1, "↻ Replay", self._replay)
        self._loop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(r1, text="Loop", variable=self._loop_var,
                       bg=_PANEL, fg=_TEXT, selectcolor=_BG,
                       activebackground=_PANEL).pack(side=tk.LEFT, padx=4)
        _btn(r1, "⊠  Reset Zoom", self._reset_zoom)
        _sep(r1)
        _lbl(r1, "Speed:")
        self._speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(r1, from_=0.25, to=4.0, variable=self._speed_var,
                  orient=tk.HORIZONTAL, length=80,
                  command=lambda v: setattr(self, "play_speed",
                                            self._speed_var.get())
                  ).pack(side=tk.LEFT)
        self._spd_lbl = tk.Label(r1, text="1.0×", bg=_PANEL, fg=_ACCENT,
                                 font=("Consolas", 8), width=4)
        self._spd_lbl.pack(side=tk.LEFT, padx=(2, 0))
        self._speed_var.trace_add("write",
            lambda *_: self._spd_lbl.configure(
                text=f"{self._speed_var.get():.1f}×"))
        _sep(r1)
        _lbl(r1, "Vid offset (s):")
        self._offset_var = tk.StringVar(value="0.0")
        oe = _ent(r1, self._offset_var, 6)
        oe.bind("<Return>", lambda e: self._apply_offset())
        _btn(r1, "Apply", self._apply_offset)
        self._vid_status = tk.Label(r1, text="No video loaded",
                                    bg=_PANEL, fg=_MUTED, font=("Segoe UI", 8))
        self._vid_status.pack(side=tk.RIGHT, padx=10)

        mpbar = tk.Frame(self.root, bg=_PANEL, pady=4, padx=8)
        mpbar.pack(side=tk.TOP, fill=tk.X)
        self._mp_btn = _btn(mpbar, "Extract Landmarks (MediaPipe)", self._extract_landmarks)
        _btn(mpbar, "Cancel extraction", self._cancel_extract.set)
        self._swap_mp = tk.BooleanVar(value=False)
        tk.Checkbutton(mpbar, text="Swap Left / Right", variable=self._swap_mp,
                       bg=_PANEL, fg=_TEXT, selectcolor=_BG).pack(side=tk.LEFT)
        self._mp_status = tk.Label(mpbar, text="Select a video, then extract", bg=_PANEL, fg=_TEXT)
        self._mp_status.pack(side=tk.LEFT, padx=10)

        r2 = tk.Frame(self.root, bg="#0c0c18", pady=5, padx=8)
        r2.pack(side=tk.TOP, fill=tk.X)

        _btn(r2, "📋  Load XLSX", self._open_xlsx)
        self._xlsx_lbl = tk.Label(r2, text="No file", bg="#0c0c18",
                                  fg=_MUTED, font=("Segoe UI", 8))
        self._xlsx_lbl.pack(side=tk.LEFT, padx=(3, 8))
        _sep(r2)

        _lbl(r2, "Analysis: full video", _ACCENT)
        _sep(r2)

        _lbl(r2, "FPS:")
        self._fps_var = tk.StringVar(value="59.97")
        _ent(r2, self._fps_var, 6)
        _sep(r2)

        _btn(r2, "⚙  Run Pipeline  (both hands)", self._run_pipeline)

        self._pipe_status = tk.Label(r2, text="Ready",
                                     bg="#0c0c18", fg=_MUTED,
                                     font=("Segoe UI", 8))
        self._pipe_status.pack(side=tk.LEFT, padx=8)

        r3 = tk.Frame(self.root, bg="#080812", pady=4, padx=8)
        r3.pack(side=tk.TOP, fill=tk.X)
        self._r3 = r3

        _lbl(r3, "Display hand:", _ACCENT)
        self._hand_var = tk.StringVar(value="Right")
        self._auto_hand = tk.BooleanVar(value=True)
        tk.Checkbutton(r3, text="Auto", variable=self._auto_hand,
            command=self._auto_changed, bg="#080812", fg=_TEXT,
            selectcolor=_BG).pack(side=tk.LEFT, padx=4)
        self._activity_status = tk.Label(r3, text="Run Pipeline for activity detection",
            bg="#080812", fg=_ACCENT, font=("Segoe UI", 9))
        self._activity_status.pack(side=tk.LEFT, padx=4)
        self._right_rb = tk.Radiobutton(
            r3, text="Right", variable=self._hand_var, value="Right",
            bg="#080812", fg=_TEXT, selectcolor=_BG,
            activebackground="#080812", font=("Segoe UI", 10, "bold"),
            command=self._switch_hand,
        )
        self._right_rb.pack(side=tk.LEFT, padx=6)
        self._left_rb = tk.Radiobutton(
            r3, text="Left", variable=self._hand_var, value="Left",
            bg="#080812", fg=_TEXT, selectcolor=_BG,
            activebackground="#080812", font=("Segoe UI", 10, "bold"),
            command=self._switch_hand,
        )
        self._left_rb.pack(side=tk.LEFT, padx=6)

        self._right_badge = tk.Label(r3, text="Right: not computed",
                                     bg="#080812", fg=_MUTED,
                                     font=("Consolas", 8), padx=8, pady=2,
                                     relief=tk.RIDGE)
        self._right_badge.pack(side=tk.LEFT, padx=8)
        self._left_badge = tk.Label(r3, text="Left: not computed",
                                    bg="#080812", fg=_MUTED,
                                    font=("Consolas", 8), padx=8, pady=2,
                                    relief=tk.RIDGE)
        self._left_badge.pack(side=tk.LEFT, padx=4)

        bot = tk.Frame(self.root, bg=_PANEL, pady=5, padx=8)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        self._tl_var = tk.IntVar(value=0)
        self._tl = ttk.Scale(bot, from_=0, to=1000,
                              variable=self._tl_var, orient=tk.HORIZONTAL,
                              command=self._on_seek)
        self._tl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._time_lbl = tk.Label(bot, text="00:00.00 / 00:00.00",
                                  bg=_PANEL, fg=_TEXT,
                                  font=("Consolas", 9), width=20)
        self._time_lbl.pack(side=tk.RIGHT, padx=8)

        main = tk.Frame(self.root, bg=_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        vf = tk.Frame(main, bg=_BG)
        vf.pack(side=tk.LEFT, fill=tk.Y)
        hdr = tk.Frame(vf, bg=_BG)
        hdr.pack(fill=tk.X, pady=(0, 3))
        tk.Label(hdr, text="VIDEO", bg=_BG, fg=_ACCENT,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr,
                 text="  drag = zoom  •  right-click = reset  •  Space = play/pause",
                 bg=_BG, fg=_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)

        self.vid_canvas = tk.Canvas(
            vf, width=self.VID_W, height=self.VID_H,
            bg="#0a0a15", highlightthickness=1,
            highlightbackground=_BORDER, cursor="crosshair"
        )
        self.vid_canvas.pack()
        self._placeholder_id = self.vid_canvas.create_text(
            self.VID_W // 2, self.VID_H // 2,
            text="📂  Open a video file",
            fill=_MUTED, font=("Segoe UI", 15),
        )
        self._zoom_lbl = tk.Label(vf, text="Zoom: full frame",
                                  bg=_BG, fg=_MUTED, font=("Consolas", 8))
        self._zoom_lbl.pack(anchor="w", pady=(3, 0))

        self.vid_canvas.bind("<ButtonPress-1>",  self._on_press)
        self.vid_canvas.bind("<B1-Motion>",       self._on_drag)
        self.vid_canvas.bind("<ButtonRelease-1>", self._on_release)
        self.vid_canvas.bind("<Button-3>",        lambda e: self._reset_zoom())
        self.vid_canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())

        wf = tk.Frame(main, bg=_BG)
        wf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        tk.Label(wf, text="WAVEFORMS  ·  Amplitude   Velocity   Speed",
                 bg=_BG, fg=_ACCENT,
                 font=("Consolas", 10, "bold")).pack(anchor="w", pady=(0, 3))
        self._wave_frame = wf

    def _build_wave_figure(self):
        fig = plt.Figure(figsize=(8.5, 7), facecolor=_BG)
        gs  = gridspec.GridSpec(3, 1, figure=fig,
                                hspace=0.42,
                                left=0.08, right=0.98,
                                top=0.95, bottom=0.06)
        self._axes = [fig.add_subplot(gs[i]) for i in range(3)]
        self._fig  = fig
        self._wave_canvas = FigureCanvasTkAgg(fig, master=self._wave_frame)
        self._wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_ax(self, ax, title, ylabel):
        ax.set_facecolor(_AX_BG)
        ax.tick_params(colors=_TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
        ax.grid(True, color=_COL_GRID, linewidth=0.6)
        ax.set_title(title, color=_TEXT, fontsize=8, pad=4)
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=7)
        ax.set_xlabel("Time  (s)", color=_MUTED, fontsize=7)

    def _draw_placeholders(self):
        specs = [("Amplitude  (normalized ratio)", "Ampl."),
                 ("Velocity  (ratio/s)",           "Vel."),
                 ("Speed  (ratio/s)",              "Spd.")]
        for ax, (ttl, ylbl) in zip(self._axes, specs):
            ax.cla()
            self._style_ax(ax, ttl, ylbl)
            ax.text(0.5, 0.5, "Run Pipeline → select hand",
                    transform=ax.transAxes, ha="center", va="center",
                    color=_MUTED, fontsize=9, style="italic")
        self._wave_canvas.draw()

    def _draw_waveforms(self):
        if self.sig_t is None:
            self._draw_placeholders()
            self._auto_changed()
            return

        tr   = self.sig_t
        t0   = float(tr[0])
        t1   = float(tr[-1])
        scale = _display_scale()
        amp  = self.sig_amp / scale
        vel  = gaussian_filter1d(self.sig_vel, sigma=3) / scale
        spd  = gaussian_filter1d(self.sig_spd, sigma=3) / scale
        hand = self._hand_var.get()

        ax_a, ax_v, ax_s = self._axes
        self._cursors = []

        def _markers(ax):
            ax.axvline(t0, color="#ffd54f", linewidth=1.2,
                       linestyle="--", alpha=0.7,
                       label=f"Start  {t0:.1f}s")
            ax.axvline(t1, color="#ff8a65", linewidth=1.2,
                       linestyle="--", alpha=0.7,
                       label=f"End  {t1:.1f}s")

        ax_a.cla()
        ax_a.plot(tr, amp, color=_COL_AMP, linewidth=0.9, label="Amplitude")
        if self.sig_tap is not None and self.sig_tap.any():
            idx = np.where(self.sig_tap)[0]
            ax_a.scatter(tr[idx], amp[idx], color=_COL_TAP, s=65, zorder=6,
                         marker="o", edgecolors="#e65100", linewidths=0.8,
                         label=f"Tap peaks  ({len(idx)})")
        _markers(ax_a)
        amp_top = max(1.05, float(np.nanmax(amp)) + 0.08)
        self._style_ax(ax_a,
                       f"Amplitude  (normalized ratio)  —  {hand} hand",
                       "Ampl.")
        ax_a.set_ylim(0, amp_top)
        ax_a.set_xlim(t0, t1)
        ax_a.legend(fontsize=6, loc="upper right",
                    labelcolor=_TEXT, facecolor=_AX_BG, edgecolor=_BORDER)

        ax_v.cla()
        ax_v.fill_between(tr, 0, vel, where=(vel >= 0),
                          color=_COL_OPEN, alpha=0.45, interpolate=True,
                          label="Opening")
        ax_v.fill_between(tr, 0, vel, where=(vel < 0),
                          color=_COL_CLOS, alpha=0.45, interpolate=True,
                          label="Closing")
        ax_v.plot(tr, vel, color="#ce93d8", linewidth=0.8, alpha=0.8)
        ax_v.axhline(0, color=_TEXT, linewidth=0.5, linestyle="--", alpha=0.3)
        _markers(ax_v)
        self._style_ax(ax_v, "Velocity  (ratio/s)", "Vel.")
        vel_lim = _sym_limit(vel)
        ax_v.set_ylim(-vel_lim, vel_lim)
        ax_v.set_xlim(t0, t1)
        ax_v.legend(fontsize=6, loc="upper right",
                    labelcolor=_TEXT, facecolor=_AX_BG, edgecolor=_BORDER)

        ax_s.cla()
        ax_s.fill_between(tr, 0, spd, color=_COL_SPD, alpha=0.45,
                          interpolate=True, label="Speed")
        ax_s.plot(tr, spd, color="#e040fb", linewidth=0.8, alpha=0.9)
        _markers(ax_s)
        self._style_ax(ax_s, "Speed  (ratio/s)", "Spd.")
        spd_top = _sym_limit(spd)
        ax_s.set_ylim(0, spd_top)
        ax_s.set_xlim(t0, t1)
        ax_s.legend(fontsize=6, loc="upper right",
                    labelcolor=_TEXT, facecolor=_AX_BG, edgecolor=_BORDER)

        for ax in self._axes:
            ln = ax.axvline(x=t0, color=_COL_CUR, linewidth=2.0,
                            zorder=12, alpha=0.9)
            self._cursors.append(ln)

        self._wave_canvas.draw()

    def _update_cursor(self, t_rel: float):
        if not self._cursors:
            return
        for ln in self._cursors:
            ln.set_xdata([t_rel, t_rel])
        self._wave_canvas.draw_idle()

    def _switch_hand(self):
        self._auto_hand.set(False)
        self._activity_status.configure(text="Manual display")
        hand = self._hand_var.get()
        data = self._hand_data.get(hand)
        if data is None:
            messagebox.showinfo(
                "No data",
                f"{hand} hand was not processed (check tapping signal or data quality).\n"
                "Run the pipeline and check tracking / tapping quality."
            )
            other = "Left" if hand == "Right" else "Right"
            if self._hand_data.get(other) is not None:
                self._hand_var.set(other)
            return
        self._load_hand_signals(hand)
        self._draw_waveforms()
        self._show_frame(self.current_fidx)

    def _auto_changed(self):
        if self._auto_hand.get():
            self._apply_auto_hand(self.current_fidx / self.src_fps - self._t_offset)
        else:
            self._activity_status.configure(text="Manual display")

    def _apply_auto_hand(self, seconds):
        if not self._auto_hand.get() or self._activity is None:
            return
        hand, label = self._activity.at(seconds)
        self._activity_status.configure(text="Auto: " + label)
        if hand is not None and self._hand_data.get(hand) is not None and self._hand_var.get()!=hand:
            self._hand_var.set(hand)
            self._load_hand_signals(hand)
            self._draw_waveforms()

    def _load_hand_signals(self, hand: str):
        d = self._hand_data[hand]
        if d is None:
            return
        self.sig_t   = d["t"]
        self.sig_amp = d["amp"]
        self.sig_vel = d["vel"]
        self.sig_spd = d["spd"]
        self.sig_tap = d["tap_mask"]

    def _open_video(self):
        if self._extracting:
            messagebox.showinfo("Extraction running", "Cancel or wait before opening another video.")
            return
        path = filedialog.askopenfilename(
            title="Open Hand Recording",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                       ("All",   "*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        self._stop_playback()
        if self.cap:
            self.cap.release()
        self._video_path = path
        self._overlay = []
        self._xlsx_path = ""
        self._xlsx_lbl.configure(text="No file")
        self._activity = None
        self._hand_data = {"Right": None, "Left": None}
        self.cap        = cap
        self.src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps_var.set(str(self.src_fps))
        self.src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_fidx = 0; self.zoom_rect = None
        self._tl.configure(to=max(1, self.src_frames - 1))
        self._suppress_seek = True
        self._tl_var.set(0)
        self._suppress_seek = False
        dur = self.src_frames / self.src_fps
        self._vid_status.configure(
            text=f"{os.path.basename(path)}  "
                 f"{self.src_w}×{self.src_h}  "
                 f"{self.src_fps:.2f} fps  {dur:.1f} s"
        )
        if self._placeholder_id:
            self.vid_canvas.delete(self._placeholder_id)
            self._placeholder_id = None
        self._show_frame(0)

    def _collect_ui_cycles(self):
        if self._closing:
            return
        gc.collect()
        self.root.after(1000, self._collect_ui_cycles)

    def _callback_error(self, exc_type, value, tb):
        details = "".join(traceback.format_exception(exc_type, value, tb))
        print(details)
        messagebox.showerror("GUI error", details, parent=self.root)

    def _extract_landmarks(self):
        if self._extracting:
            return
        if not self._video_path:
            messagebox.showinfo("No video", "Open a video first.")
            return
        self._stop_playback()
        folder = os.path.dirname(os.path.abspath(__file__))
        model = os.path.join(folder, "hand_landmarker.task")
        if not os.path.isfile(model):
            messagebox.showerror("Model missing", "Place hand_landmarker.task next to this script.", parent=self.root)
            return
        from datetime import datetime
        output_dir = os.path.join(folder, "landmark_exports")
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self._video_path))[0]
        output = os.path.join(output_dir, stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_landmarks.xlsx")
        self._extracting = True
        self._exit_seen = False
        self._cancel_extract.clear()
        self._mp_btn.configure(state=tk.DISABLED)
        self._mp_status.configure(text="Starting MediaPipe …")
        video, swap = self._video_path, self._swap_mp.get()
        self._extract_process = self._mp_context.Process(target=extraction_worker,
            args=(video, model, output, swap, self._extract_queue, self._cancel_extract), daemon=True)
        # Give Tk a chance to paint the status before starting the child process.
        self.root.after(50, self._start_extraction_process)

    def _start_extraction_process(self):
        try:
            self._extract_process.start()
        except Exception:
            self._extracting = False
            self._mp_btn.configure(state=tk.NORMAL)
            self._mp_status.configure(text="Could not start extraction")
            messagebox.showerror("Extraction error", traceback.format_exc(), parent=self.root)
            return
        self.root.after(100, self._poll_extraction)

    def _poll_extraction(self):
        try:
            while True:
                kind, payload = self._extract_queue.get_nowait()
                if kind == "progress":
                    n,total = payload
                    self._mp_status.configure(text=f"Processing {n} / {total or '?'} frames")
                    continue
                self._extracting = False
                self._mp_btn.configure(state=tk.NORMAL)
                if kind == "done":
                    path, fps, self._overlay = payload
                    self._xlsx_path = path
                    self._activity = None
                    self._hand_data = {"Right": None, "Left": None}
                    self.sig_t = None
                    self._draw_placeholders()
                    self._activity_status.configure(text="Run Pipeline for activity detection")
                    self._fps_var.set(str(fps))
                    self._xlsx_lbl.configure(text=os.path.basename(path), fg=_COL_TAP)
                    self._mp_status.configure(text="Extracted. Run Pipeline to analyse the full video.")
                    self._show_frame(self.current_fidx)
                elif kind == "cancelled":
                    self._mp_status.configure(text="Extraction cancelled")
                else:
                    self._mp_status.configure(text="Extraction failed")
                    messagebox.showerror("MediaPipe error", payload)
        except queue.Empty:
            pass
        if self._extracting:
            if self._extract_process is not None and self._extract_process.exitcode is not None:
                # Allow the IPC queue feeder to deliver the terminal message first.
                if not getattr(self, "_exit_seen", False):
                    self._exit_seen = True
                    self.root.after(300, self._poll_extraction)
                    return
                self._extracting = False
                self._mp_btn.configure(state=tk.NORMAL)
                self._mp_status.configure(text="Extraction process exited unexpectedly")
                messagebox.showerror("Extraction process stopped",
                    f"Exit code: {self._extract_process.exitcode}. Check terminal output.", parent=self.root)
                return
            self.root.after(100, self._poll_extraction)

    def _open_xlsx(self):
        if self._extracting:
            return
        path = filedialog.askopenfilename(
            title="Load Landmark XLSX",
            filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")]
        )
        if not path:
            return
        self._xlsx_path = path
        self._activity = None
        self._hand_data = {"Right": None, "Left": None}
        self.sig_t = None
        self._draw_placeholders()
        self._activity_status.configure(text="Run Pipeline for activity detection")
        self._overlay = []
        self._xlsx_lbl.configure(text=os.path.basename(path), fg=_COL_TAP)
        self._pipe_status.configure(
            text="XLSX loaded → Run Pipeline (full recording)",
            fg=_MUTED
        )

    def _run_pipeline(self):
        if self._extracting:
            messagebox.showinfo("Extraction running", "Wait for landmark extraction to finish.")
            return
        if not _PIPELINE_OK:
            messagebox.showerror(
                "Pipeline unavailable",
                f"Import failed:\n{_PIPELINE_ERR}\n\n"
                "Place correct_tap_detection.py and feature_extraction.py next to this script."
            )
            return
        if not self._xlsx_path:
            messagebox.showwarning("No XLSX", "Load an XLSX file first.")
            return

        try:
            fps     = float(self._fps_var.get())
        except ValueError:
            messagebox.showerror("Input error",
                                 "FPS must be a positive number.")
            return

        self._pipe_status.configure(text="⏳  Running pipeline …", fg=_COL_TAP)
        self._right_badge.configure(text="Right: computing …", fg=_COL_TAP)
        self._left_badge.configure( text="Left:  computing …", fg=_COL_TAP)
        self.root.update_idletasks()


        try:
            extractor = TapExtractor(
                self._xlsx_path,
                fps         = fps,
            )
            extractor.load_excel_dataset()
            extractor.compute_distances()
            results = extractor.tap_detection()
            self._activity = extractor.activity

        except Exception:
            tb = traceback.format_exc()
            messagebox.showerror("TapExtractor error", tb)
            self._pipe_status.configure(text="Error — see dialog", fg=_ERR_RED)
            return

        fe = FeatureExtractor()
        summary_parts = []

        for hand, t_arr, amp_arr in [
            ("Right", extractor.right_hand_time, extractor.right_dist_normalized),
            ("Left",  extractor.left_hand_time,  extractor.left_dist_normalized),
        ]:
            badge = self._right_badge if hand == "Right" else self._left_badge
            try:
                hf = extract_hand_features(
                    extractor, results, fe, hand,
                    fs    = fps,
                )
            except Exception:
                tb = traceback.format_exc()
                print(f"\n[{hand}] extract_hand_features error:\n{tb}")
                hf = None

            if hf is None:
                self._hand_data[hand] = None
                badge.configure(
                    text=f"{hand}: skipped — insufficient taps or missing data",
                    fg=_ERR_RED
                )
                summary_parts.append(f"{hand}: FAILED")
                continue

            peaks    = hf["tap_features"]["validated_peaks"]
            tap_mask = np.zeros(len(t_arr), dtype=bool)
            for pk in peaks:
                pk = int(pk)
                if 0 <= pk < len(t_arr):
                    tap_mask[pk] = True

            self._hand_data[hand] = dict(
                t        = np.asarray(t_arr,               dtype=float),
                amp      = np.asarray(amp_arr,             dtype=float),
                vel      = np.asarray(hf["velocity_signal"],dtype=float),
                spd      = np.asarray(hf["speed_signal"],  dtype=float),
                tap_mask = tap_mask,
                hf       = hf,
            )

            n_taps  = int(tap_mask.sum())
            dur     = float(t_arr[-1] - t_arr[0])
            cd      = hf["cycle_data"]["mean_cycle_dur"]
            amp_pct = hf["hb_features"]["amp_avg_pct"]
            cv      = hf["cycle_data"]["cv_iti"]
            badge.configure(
                text=(f"{hand}: ✓  {n_taps} taps  {dur:.1f}s  "
                      f"Amp={amp_pct:.1f}%  CD={cd:.3f}s  CV={cv:.3f}"),
                fg=_OK_GREEN
            )
            summary_parts.append(
                f"{hand}: {n_taps} taps  Amp={amp_pct:.1f}%  CV={cv:.3f}"
            )

        plt.close("all")

        current = self._hand_var.get()
        if self._hand_data[current] is not None:
            chosen = current
        elif self._hand_data["Right"] is not None:
            chosen = "Right"
        elif self._hand_data["Left"] is not None:
            chosen = "Left"
        else:
            self._pipe_status.configure(
                text="No analysable taps — check detection quality and tapping signal", fg=_ERR_RED)
            self._draw_placeholders()
            self._auto_changed()
            return

        self._hand_var.set(chosen)
        self._load_hand_signals(chosen)
        self._draw_waveforms()
        self._show_frame(self.current_fidx)

        self._pipe_status.configure(
            text="✓ Full video | " + "   |   ".join(summary_parts),
            fg=_OK_GREEN
        )

    def _disp_to_src(self, dx, dy):
        cw, ch = self.VID_W, self.VID_H
        if self.zoom_rect:
            zx1, zy1, zx2, zy2 = self.zoom_rect
            zw, zh = zx2-zx1, zy2-zy1
            sc = min(cw/zw, ch/zh)
            ox = (cw - int(zw*sc))//2
            oy = (ch - int(zh*sc))//2
            sx = zx1 + (dx-ox)/sc
            sy = zy1 + (dy-oy)/sc
        else:
            sw, sh = self.src_w, self.src_h
            sc = min(cw/sw, ch/sh) if sw and sh else 1.0
            ox = (cw - int(sw*sc))//2
            oy = (ch - int(sh*sc))//2
            sx = (dx-ox)/sc
            sy = (dy-oy)/sc
        return (float(np.clip(sx, 0, self.src_w-1)),
                float(np.clip(sy, 0, self.src_h-1)))

    def _on_press(self, e):
        self._drag_start = (e.x, e.y)
        if self._rect_id:
            self.vid_canvas.delete(self._rect_id); self._rect_id = None

    def _on_drag(self, e):
        if not self._drag_start: return
        x0, y0 = self._drag_start
        if self._rect_id: self.vid_canvas.delete(self._rect_id)
        self._rect_id = self.vid_canvas.create_rectangle(
            x0, y0, e.x, e.y, outline=_COL_TAP, width=2, dash=(5,3))

    def _on_release(self, e):
        if not self._drag_start: return
        x0, y0 = self._drag_start
        x1, y1 = e.x, e.y
        self._drag_start = None
        if self._rect_id:
            self.vid_canvas.delete(self._rect_id); self._rect_id = None
        if abs(x1-x0) < 12 or abs(y1-y0) < 12: return
        sx0, sy0 = self._disp_to_src(min(x0,x1), min(y0,y1))
        sx1, sy1 = self._disp_to_src(max(x0,x1), max(y0,y1))
        if (sx1-sx0) < 20 or (sy1-sy0) < 20: return
        self.zoom_rect = (int(sx0), int(sy0), int(sx1), int(sy1))
        self._show_frame(self.current_fidx)

    def _reset_zoom(self):
        self.zoom_rect = None
        self._zoom_lbl.configure(text="Zoom: full frame")
        self._show_frame(self.current_fidx)

    def _read_frame(self, fidx):
        if not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = self.cap.read()
        return fr if ok else None

    def _show_frame(self, fidx: int):
        if not self.cap: return
        fidx = int(np.clip(fidx, 0, self.src_frames-1))
        raw  = self._read_frame(fidx)
        if raw is None: return
        self._last_raw = raw; self.current_fidx = fidx

        if fidx < len(self._overlay):
            raw = draw_overlay(raw, self._overlay[fidx])
        disp = raw
        if self.zoom_rect:
            x1, y1, x2, y2 = self.zoom_rect
            x1=max(0,min(x1,self.src_w-1)); x2=max(x1+2,min(x2,self.src_w))
            y1=max(0,min(y1,self.src_h-1)); y2=max(y1+2,min(y2,self.src_h))
            disp = raw[y1:y2, x1:x2]
            self._zoom_lbl.configure(
                text=f"Zoom  {x1},{y1}→{x2},{y2}  ({x2-x1}×{y2-y1} px)")

        cw, ch = self.VID_W, self.VID_H
        h, w   = disp.shape[:2]
        sc = min(cw/w, ch/h)
        nw, nh = max(1,int(w*sc)), max(1,int(h*sc))
        sized  = cv2.resize(disp, (nw,nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((ch,cw,3), dtype=np.uint8)
        ox, oy = (cw-nw)//2, (ch-nh)//2
        canvas[oy:oy+nh, ox:ox+nw] = sized

        if self.zoom_rect and self._last_raw is not None:
            MW, MH = 130, 98
            mini = cv2.resize(self._last_raw, (MW,MH))
            zx1,zy1,zx2,zy2 = self.zoom_rect
            cv2.rectangle(mini,
                (int(zx1/self.src_w*MW), int(zy1/self.src_h*MH)),
                (int(zx2/self.src_w*MW), int(zy2/self.src_h*MH)),
                (255,213,79), 1)
            cv2.rectangle(mini,(0,0),(MW-1,MH-1),(80,100,200),1)
            cv2.putText(mini,"overview",(3,MH-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,(160,160,210),1,cv2.LINE_AA)
            canvas[4:4+MH, cw-MW-4:cw-4] = mini

        rgb   = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.vid_canvas.create_image(0,0,anchor=tk.NW,image=photo)
        self._photo = photo

        t_abs = fidx / self.src_fps
        self._apply_auto_hand(t_abs - self._t_offset)
        dur   = self.src_frames / self.src_fps
        self._time_lbl.configure(text=f"{_fmt(t_abs)} / {_fmt(dur)}")

        self._suppress_seek = True
        self._tl_var.set(int(fidx / max(1,self.src_frames-1) * 1000))
        self._suppress_seek = False

        if self.sig_t is not None:
            t_cur = t_abs - self._t_offset
            self._update_cursor(t_cur)

    def _toggle_play(self):
        if not self.cap: return
        self._stop_playback() if self.playing else self._start_playback()

    def _start_playback(self):
        if not self.cap or self.src_frames <= 0 or self._closing:
            return
        if self.playing:
            return
        if self.current_fidx >= self.src_frames - 1:
            self._show_frame(0)
        self.playing = True
        self._play_btn.configure(text="⏸  Pause")
        self._schedule_tick()

    def _replay(self):
        if not self.cap or self.src_frames <= 0:
            return
        self._stop_playback()
        self._show_frame(0)
        self._start_playback()

    def _schedule_tick(self):
        delay = max(1, int(1000 / self.src_fps / max(0.1, self.play_speed)))
        self._after_id = self.root.after(delay, self._tick)

    def _stop_playback(self):
        self.playing = False
        self._play_btn.configure(text="▶  Play")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self):
        self._after_id = None
        if not self.playing or self._closing:
            return
        nxt = self.current_fidx + 1
        if nxt >= self.src_frames:
            if self._loop_var.get():
                nxt = 0
            else:
                self._stop_playback()
                return
        self._show_frame(nxt)
        self._schedule_tick()

    def _step(self, d):
        if not self.cap: return
        self._stop_playback(); self._show_frame(self.current_fidx + d)

    def _on_seek(self, _=None):
        if not self.cap or self._suppress_seek:
            return
        fidx = int(self._tl_var.get()/1000*(self.src_frames-1))
        was  = self.playing
        self._stop_playback(); self._show_frame(fidx)
        if was: self._start_playback()

    def _apply_offset(self):
        try:
            self._t_offset = float(self._offset_var.get())
        except ValueError:
            return
        self._show_frame(self.current_fidx)

    def _on_close(self):
        if self._closing:
            return
        self._closing = True
        self._cancel_extract.set()
        self._stop_playback()
        process = self._extract_process
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.5)
        # No timer may restart playback or poll after destroying Tcl.
        for timer in self.root.tk.call("after", "info"):
            self.root.after_cancel(timer)
        if self.cap:
            self.cap.release()
        plt.close("all")
        self._photo = None
        gc.collect()
        self.root.destroy()
        self._extract_queue.close()
        self._extract_queue.cancel_join_thread()



if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = HandSyncViewer(root)
    try:
        root.mainloop()
    finally:
        if not app._closing:
            app._on_close()
        del app
        del root
        gc.collect()