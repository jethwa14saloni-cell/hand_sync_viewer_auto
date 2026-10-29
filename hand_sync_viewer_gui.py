import gc
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter1d

from mediapipe_extraction import extraction_worker, draw_overlay

try:
    import inspect
    import correct_tap_detection
    import feature_extraction
    from correct_tap_detection import TapExtractor
    from feature_extraction import FeatureExtractor, extract_hand_features
    _PIPELINE_OK = True
    _PIPELINE_ERR = ""
    _stale = []
    if "t_start" not in inspect.signature(TapExtractor.__init__).parameters:
        _stale.append(correct_tap_detection.__file__)
    if "_cycle_durations" not in vars(FeatureExtractor):
        _stale.append(feature_extraction.__file__)
    if _stale:
        _PIPELINE_OK = False
        _PIPELINE_ERR = ("These files are an older version than this GUI:\n"
                         + "\n".join(_stale)
                         + "\n\nReplace them with the new correct_tap_detection.py, "
                           "feature_extraction.py and hand_activity.py.")
except ImportError as _ie:
    _PIPELINE_OK = False
    _PIPELINE_ERR = (f"Could not import the pipeline:\n{_ie}\n\n"
                     "Place correct_tap_detection.py, feature_extraction.py and "
                     "hand_activity.py next to this script.")

_BG = "#0e0e1a"
_PANEL = "#13131f"
_AX_BG = "#191928"
_BORDER = "#2a2a45"
_TEXT = "#c8c8e0"
_MUTED = "#555577"
_ACCENT = "#4fc3f7"

_COL_AMP = "#4fc3f7"
_COL_OPEN = "#42a5f5"
_COL_CLOS = "#ef5350"
_COL_SPD = "#ab47bc"
_COL_TAP = "#ffd54f"
_COL_CUR = "#ff5252"
_COL_ACT = "#81c784"
_COL_GRID = "#1e1e32"
_COL_OUT = "#05050c"
_WARN = "#ffb74d"
_OK_GREEN = "#81c784"
_ERR_RED = "#ef9a9a"

_NO_ACTIVITY = "Run Pipeline for activity detection"


def _fmt(s):
    s = max(0.0, s)
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def _sym_limit(x, pad=1.1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    m = float(np.max(np.abs(x)))
    return m * pad if m > 0 else 1.0


def _runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _xlsx_fps(path):
    try:
        meta = pd.read_excel(path, sheet_name="metadata")
        fps = float(meta["fps"].iloc[0])
    except Exception:
        return None
    return fps if np.isfinite(fps) and fps > 0 else None


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _find_model():
    places = [_app_dir(), getattr(sys, "_MEIPASS", "")]
    for folder in places:
        path = os.path.join(folder, "hand_landmarker.task")
        if folder and os.path.isfile(path):
            return path
    return None


def _output_dir():
    candidates = [os.path.join(_app_dir(), "landmark_exports"),
                  os.path.join(os.path.expanduser("~"), "Documents", "HandSyncViewer", "landmark_exports")]
    for folder in candidates:
        try:
            os.makedirs(folder, exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w"):
                pass
            os.remove(probe)
            return folder
        except OSError:
            continue
    raise OSError("No writable folder for landmark exports.")


def _is_text_input(widget):
    return isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox))


def _pipeline_job(xlsx_path, fps, t_start, t_end, out):
    try:
        extractor = TapExtractor(xlsx_path, fps=fps, t_start=t_start, t_end=t_end)
        extractor.load_excel_dataset()
        extractor.compute_distances()
        results = extractor.tap_detection()
        fe = FeatureExtractor()
        window = extractor.window_mask
        hands = {}
        for hand, t_arr, amp_arr, raw in (
            ("Right", extractor.right_hand_time, extractor.right_dist_normalized, extractor.right_dist_raw),
            ("Left", extractor.left_hand_time, extractor.left_dist_normalized, extractor.left_dist_raw),
        ):
            t_arr = np.asarray(t_arr, dtype=float)
            amp_arr = np.asarray(amp_arr, dtype=float)
            vel, _ = fe.compute_velocity_acceleration(amp_arr, t_arr)
            spd = fe.compute_speed(amp_arr, t_arr)
            column = f"{hand.lower()}_hand_detected"
            if column in extractor.df:
                detected = extractor.df[column].fillna(False).to_numpy(dtype=bool)
            else:
                detected = np.isfinite(np.asarray(raw, dtype=float))
            coverage = float(detected[window].mean()) if window.any() else 0.0
            reason = results.get(hand, {}).get("reason", "")
            try:
                hf = extract_hand_features(extractor, results, fe, hand, fs=fps)
            except Exception:
                print(f"\n[{hand}] extract_hand_features error:\n{traceback.format_exc()}")
                hf = None
                reason = reason or "feature extraction failed (see terminal)"
            tap_mask = np.zeros(len(t_arr), dtype=bool)
            if hf is not None:
                peaks = np.asarray(hf["tap_features"]["validated_peaks"], dtype=int)
                tap_mask[peaks[(peaks >= 0) & (peaks < len(t_arr))]] = True
            elif not reason:
                reason = "fewer than 3 taps"
            hands[hand] = dict(t=t_arr, amp=amp_arr, vel=np.asarray(vel, dtype=float),
                               spd=np.asarray(spd, dtype=float), tap_mask=tap_mask, hf=hf,
                               reason=reason, coverage=coverage)
        w = extractor.time[window]
        out.put(("done", (extractor.activity, hands, (float(w[0]), float(w[-1])))))
    except Exception:
        out.put(("error", traceback.format_exc()))


class HandSyncViewer:
    VID_W = 640
    VID_H = 480

    def __init__(self, root):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("HandSyncViewer must be created on the main thread")
        gc.disable()
        self._closing = False
        self.root = root
        self.root.report_callback_exception = self._callback_error
        self.root.title("Hand Sync Viewer")
        self.root.configure(bg=_BG)
        self.root.minsize(1180, 680)

        self._video_path = ""
        self._overlay = []
        self._extracting = False
        self._exit_seen = False
        self._mp_context = multiprocessing.get_context("spawn")
        self._cancel_extract = self._mp_context.Event()
        self._extract_queue = None
        self._extract_process = None

        self._session = 0
        self._pipeline_busy = False

        self.cap = None
        self._cap_pos = None
        self.src_fps = 30.0
        self.src_frames = 0
        self.src_w = self.VID_W
        self.src_h = self.VID_H
        self.current_fidx = 0
        self.playing = False
        self.play_speed = 1.0
        self._after_id = None
        self._play_t0 = 0.0
        self._play_f0 = 0
        self._last_raw = None
        self._last_raw_idx = -1
        self._photo = None
        self._img_id = None
        self._placeholder_id = None
        self._suppress_seek = False
        self._cw = self.VID_W
        self._ch = self.VID_H
        self._resize_job = None

        self.zoom_rect = None
        self._drag_start = None
        self._rect_id = None

        self._activity = None
        self._activity_label = ""
        self._hand_data = {"Right": None, "Left": None}
        self._shown_hand = None

        self.sig_t = None
        self.sig_amp = None
        self.sig_vel = None
        self.sig_spd = None
        self.sig_tap = None
        self._t_offset = 0.0

        self._fig = None
        self._axes = []
        self._cursors = []
        self._wave_canvas = None
        self._wave_bg = None
        self._view_width = 10.0
        self._view_x0 = 0.0
        self._last_scroll = 0.0

        self._xlsx_path = ""
        self._analysis_window = None

        self._setup_style()
        self._build_ui()
        self._build_wave_figure()
        self._draw_placeholders()

        bindings = (
            ("<space>", self._toggle_play),
            ("<Right>", lambda: self._step(+1)),
            ("<Left>", lambda: self._step(-1)),
            ("<Shift-Right>", lambda: self._step(+10)),
            ("<Shift-Left>", lambda: self._step(-10)),
            ("<Home>", lambda: self._seek_frame(0)),
            ("<End>", lambda: self._seek_frame(self.src_frames - 1)),
            ("r", self._reset_zoom),
            ("R", self._reset_zoom),
            ("q", self._on_close),
            ("<Escape>", self._on_close),
        )
        for seq, fn in bindings:
            self.root.bind(seq, self._key(fn))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(1000, self._collect_ui_cycles)

        if not _PIPELINE_OK:
            messagebox.showwarning("Pipeline modules", _PIPELINE_ERR)

    def _key(self, fn):
        def handler(event):
            if _is_text_input(event.widget):
                return None
            fn()
            return "break"
        return handler

    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Horizontal.TScale", background=_PANEL,
                        troughcolor=_BORDER, bordercolor=_BORDER,
                        lightcolor=_ACCENT, darkcolor=_ACCENT)
        style.configure("Dark.TCombobox", fieldbackground=_BORDER, background=_BORDER,
                        foreground=_TEXT, arrowcolor=_TEXT, bordercolor=_BORDER)
        style.map("Dark.TCombobox", fieldbackground=[("readonly", _BORDER)],
                  foreground=[("readonly", _TEXT)])

    def _build_ui(self):

        def _btn(p, text, cmd, w=None, state=tk.NORMAL):
            kw = {"width": w} if w else {}
            b = tk.Button(p, text=text, command=cmd,
                          bg=_BORDER, fg=_TEXT, relief=tk.FLAT,
                          activebackground="#3a3a60", activeforeground="white",
                          disabledforeground=_MUTED,
                          padx=9, pady=4, font=("Segoe UI", 9),
                          cursor="hand2", takefocus=0, state=state, **kw)
            b.pack(side=tk.LEFT, padx=2)
            return b

        def _chk(p, text, var, cmd=None):
            c = tk.Checkbutton(p, text=text, variable=var, command=cmd,
                               bg=p["bg"], fg=_TEXT, selectcolor=_BG,
                               activebackground=p["bg"], activeforeground=_TEXT,
                               takefocus=0)
            c.pack(side=tk.LEFT, padx=4)
            return c

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

        def _unfocus(widget):
            widget.bind("<ButtonRelease-1>", lambda e: self.root.focus_set(), add="+")

        r1 = tk.Frame(self.root, bg=_PANEL, pady=5, padx=8)
        r1.pack(side=tk.TOP, fill=tk.X)

        _btn(r1, "📂  Open Video", self._open_video)
        _sep(r1)
        self._play_btn = _btn(r1, "▶  Play", self._toggle_play, w=10)
        _btn(r1, "↻ Replay", self._replay)
        self._loop_var = tk.BooleanVar(value=True)
        _chk(r1, "Loop", self._loop_var)
        _btn(r1, "⊠  Reset Zoom", self._reset_zoom)
        _sep(r1)
        _lbl(r1, "Speed:")
        self._speed_var = tk.DoubleVar(value=1.0)
        spd_scale = ttk.Scale(r1, from_=0.25, to=4.0, variable=self._speed_var,
                              orient=tk.HORIZONTAL, length=90,
                              command=self._set_speed, takefocus=0)
        spd_scale.pack(side=tk.LEFT)
        _unfocus(spd_scale)
        self._spd_lbl = tk.Label(r1, text="1.00×", bg=_PANEL, fg=_ACCENT,
                                 font=("Consolas", 8), width=5)
        self._spd_lbl.pack(side=tk.LEFT, padx=(2, 0))
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
        self._cancel_btn = _btn(mpbar, "Cancel extraction", self._cancel_extraction,
                                state=tk.DISABLED)
        self._swap_mp = tk.BooleanVar(value=False)
        _chk(mpbar, "Swap Left / Right", self._swap_mp)
        self._mp_status = tk.Label(mpbar, text="Select a video, then extract",
                                   bg=_PANEL, fg=_TEXT, font=("Segoe UI", 9))
        self._mp_status.pack(side=tk.LEFT, padx=10)

        r2 = tk.Frame(self.root, bg="#0c0c18", pady=5, padx=8)
        r2.pack(side=tk.TOP, fill=tk.X)

        _btn(r2, "📋  Load XLSX", self._open_xlsx)
        self._xlsx_lbl = tk.Label(r2, text="No file", bg="#0c0c18",
                                  fg=_MUTED, font=("Segoe UI", 8))
        self._xlsx_lbl.pack(side=tk.LEFT, padx=(3, 8))
        _sep(r2)
        _lbl(r2, "Analyse from", _ACCENT)
        self._win_start_var = tk.StringVar(value="")
        ws = _ent(r2, self._win_start_var, 6)
        _btn(r2, "⇤ cursor", lambda: self._set_window_edge(self._win_start_var))
        _lbl(r2, "to", _ACCENT)
        self._win_end_var = tk.StringVar(value="")
        we = _ent(r2, self._win_end_var, 6)
        _btn(r2, "cursor ⇥", lambda: self._set_window_edge(self._win_end_var))
        _btn(r2, "Full", self._clear_window)
        for e in (ws, we):
            e.bind("<Return>", lambda ev: self._run_pipeline())
        _sep(r2)
        _lbl(r2, "FPS:")
        self._fps_var = tk.StringVar(value="59.97")
        _ent(r2, self._fps_var, 8)
        _sep(r2)
        self._pipe_btn = _btn(r2, "⚙  Run Pipeline  (both hands)", self._run_pipeline)
        self._pipe_status = tk.Label(r2, text="Ready", bg="#0c0c18",
                                     fg=_MUTED, font=("Segoe UI", 8))
        self._pipe_status.pack(side=tk.LEFT, padx=8)

        r3 = tk.Frame(self.root, bg="#080812", pady=4, padx=8)
        r3.pack(side=tk.TOP, fill=tk.X)

        _lbl(r3, "Display hand:", _ACCENT)
        self._hand_var = tk.StringVar(value="Right")
        self._auto_hand = tk.BooleanVar(value=True)
        _chk(r3, "Auto", self._auto_hand, self._auto_changed)
        self._activity_status = tk.Label(r3, text=_NO_ACTIVITY, bg="#080812",
                                         fg=_ACCENT, font=("Segoe UI", 9),
                                         width=28, anchor="w")
        self._activity_status.pack(side=tk.LEFT, padx=4)
        self._activity_label = _NO_ACTIVITY
        for hand in ("Right", "Left"):
            tk.Radiobutton(
                r3, text=hand, variable=self._hand_var, value=hand,
                bg="#080812", fg=_TEXT, selectcolor=_BG,
                activebackground="#080812", activeforeground=_TEXT,
                font=("Segoe UI", 10, "bold"), takefocus=0,
                command=self._switch_hand,
            ).pack(side=tk.LEFT, padx=6)

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
        self._tl_var = tk.DoubleVar(value=0)
        self._tl = ttk.Scale(bot, from_=0, to=1, variable=self._tl_var,
                             orient=tk.HORIZONTAL, command=self._on_seek,
                             takefocus=0)
        self._tl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        _unfocus(self._tl)
        self._time_lbl = tk.Label(bot, text="00:00.00 / 00:00.00",
                                  bg=_PANEL, fg=_TEXT,
                                  font=("Consolas", 9), width=20)
        self._time_lbl.pack(side=tk.RIGHT, padx=8)

        main = tk.Frame(self.root, bg=_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        vf = tk.Frame(main, bg=_BG)
        vf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hdr = tk.Frame(vf, bg=_BG)
        hdr.pack(fill=tk.X, pady=(0, 3))
        tk.Label(hdr, text="VIDEO", bg=_BG, fg=_ACCENT,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr,
                 text="  drag = zoom  •  right-click = reset  •  Space = play/pause  •  ←/→ = frame",
                 bg=_BG, fg=_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)

        border = tk.Frame(vf, bg=_BORDER, padx=1, pady=1)
        border.pack(fill=tk.BOTH, expand=True)
        self.vid_canvas = tk.Canvas(border, width=self.VID_W, height=self.VID_H,
                                    bg="#0a0a15", highlightthickness=0,
                                    cursor="crosshair")
        self.vid_canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self.vid_canvas.create_image(0, 0, anchor=tk.NW)
        self._placeholder_id = self.vid_canvas.create_text(
            self.VID_W // 2, self.VID_H // 2,
            text="📂  Open a video file", fill=_MUTED, font=("Segoe UI", 15),
        )
        self._zoom_lbl = tk.Label(vf, text="Zoom: full frame",
                                  bg=_BG, fg=_MUTED, font=("Consolas", 8))
        self._zoom_lbl.pack(anchor="w", pady=(3, 0))

        self.vid_canvas.bind("<Configure>", self._on_canvas_resize)
        self.vid_canvas.bind("<ButtonPress-1>", self._on_press)
        self.vid_canvas.bind("<B1-Motion>", self._on_drag)
        self.vid_canvas.bind("<ButtonRelease-1>", self._on_release)
        self.vid_canvas.bind("<Button-3>", lambda e: self._reset_zoom())
        self.vid_canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())

        wf = tk.Frame(main, bg=_BG)
        wf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        whdr = tk.Frame(wf, bg=_BG)
        whdr.pack(fill=tk.X, pady=(0, 3))
        tk.Label(whdr, text="WAVEFORMS", bg=_BG, fg=_ACCENT,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(whdr, text="  click = seek  •  wheel = zoom  •  Shift+wheel = scroll",
                 bg=_BG, fg=_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._follow_var = tk.BooleanVar(value=True)
        tk.Checkbutton(whdr, text="Follow video", variable=self._follow_var,
                       command=self._on_follow_toggle, bg=_BG, fg=_TEXT,
                       selectcolor=_BG, activebackground=_BG, activeforeground=_TEXT,
                       takefocus=0).pack(side=tk.RIGHT, padx=4)
        self._view_var = tk.StringVar(value="10 s")
        view_box = ttk.Combobox(whdr, textvariable=self._view_var, width=6,
                                values=("3 s", "5 s", "10 s", "20 s", "Full"),
                                state="readonly", style="Dark.TCombobox", takefocus=0)
        view_box.pack(side=tk.RIGHT, padx=(2, 6))
        view_box.bind("<<ComboboxSelected>>", self._on_view_selected)
        tk.Label(whdr, text="Show:", bg=_BG, fg=_MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT)
        self._wave_frame = wf

    def _build_wave_figure(self):
        fig = Figure(figsize=(8.5, 7), facecolor=_BG)
        gs = fig.add_gridspec(3, 1, hspace=0.45, left=0.09, right=0.98,
                              top=0.95, bottom=0.07)
        self._axes = [fig.add_subplot(gs[i]) for i in range(3)]
        self._fig = fig
        self._wave_canvas = FigureCanvasTkAgg(fig, master=self._wave_frame)
        widget = self._wave_canvas.get_tk_widget()
        widget.configure(bg=_BG, highlightthickness=0)
        widget.pack(fill=tk.BOTH, expand=True)
        self._wave_canvas.mpl_connect("draw_event", self._on_wave_draw)
        self._wave_canvas.mpl_connect("resize_event", self._on_wave_resize)
        self._wave_canvas.mpl_connect("button_press_event", self._on_wave_click)
        self._wave_canvas.mpl_connect("scroll_event", self._on_wave_scroll)

    def _style_ax(self, ax, title, ylabel):
        ax.set_facecolor(_AX_BG)
        ax.tick_params(colors=_TEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)
        ax.grid(True, color=_COL_GRID, linewidth=0.6)
        ax.set_title(title, color=_TEXT, fontsize=8, pad=4)
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=7)
        ax.set_xlabel("Time  (s)", color=_MUTED, fontsize=7)

    def _legend(self, ax):
        ax.legend(fontsize=6, loc="upper right", labelcolor=_TEXT,
                  facecolor=_AX_BG, edgecolor=_BORDER)

    def _draw_placeholders(self):
        self._cursors = []
        specs = [("Amplitude  (normalized ratio)", "Ampl."),
                 ("Velocity  (ratio/s)", "Vel."),
                 ("Speed  (ratio/s)", "Spd.")]
        for ax, (ttl, ylbl) in zip(self._axes, specs):
            ax.cla()
            self._style_ax(ax, ttl, ylbl)
            ax.text(0.5, 0.5, "Run Pipeline → select hand",
                    transform=ax.transAxes, ha="center", va="center",
                    color=_MUTED, fontsize=9, style="italic")
        self._wave_canvas.draw()

    def _activity_spans(self, hand):
        act = self._activity
        if act is None:
            return []
        scores = act.scores.get(hand)
        times = np.asarray(act.times, dtype=float)
        if scores is None or times.size == 0:
            return []
        half = (times[1] - times[0]) / 2 if times.size > 1 else 0.125
        return [(times[a] - half, times[b] + half)
                for a, b in _runs(np.asarray(scores) > 0)]

    def _draw_waveforms(self):
        if self.sig_t is None:
            self._draw_placeholders()
            return

        tr = self.sig_t
        t0 = float(tr[0])
        t1 = float(tr[-1])
        amp = self.sig_amp
        vel = gaussian_filter1d(self.sig_vel, sigma=3)
        spd = gaussian_filter1d(self.sig_spd, sigma=3)
        hand = self._hand_var.get()
        spans = self._activity_spans(hand)

        ax_a, ax_v, ax_s = self._axes
        for ax in self._axes:
            ax.cla()
            if self._analysis_window is not None:
                w0, w1 = self._analysis_window
                if w0 > t0:
                    ax.axvspan(t0, w0, color=_COL_OUT, alpha=0.6, linewidth=0, zorder=0.5)
                if w1 < t1:
                    ax.axvspan(w1, t1, color=_COL_OUT, alpha=0.6, linewidth=0, zorder=0.5)
            for i, (lo, hi) in enumerate(spans):
                label = "Tapping activity" if (ax is ax_a and i == 0) else "_nolegend_"
                ax.axvspan(lo, hi, color=_COL_ACT, alpha=0.10, linewidth=0, label=label)

        ax_a.plot(tr, amp, color=_COL_AMP, linewidth=0.9, label="Amplitude")
        if self.sig_tap is not None and self.sig_tap.any():
            idx = np.flatnonzero(self.sig_tap)
            ax_a.scatter(tr[idx], amp[idx], color=_COL_TAP, s=65, zorder=6,
                         marker="o", edgecolors="#e65100", linewidths=0.8,
                         label=f"Tap peaks  ({len(idx)})")
        amp_max = float(np.nanmax(amp)) if np.any(np.isfinite(amp)) else 1.0
        d = self._hand_data.get(hand) or {}
        note = "" if d.get("hf") is not None else f"   [no taps: {d.get('reason', '')}]"
        self._style_ax(ax_a, f"Amplitude  (normalized ratio)  —  {hand} hand"
                             f"  ·  tracked {100 * d.get('coverage', 0):.0f}%{note}", "Ampl.")
        ax_a.set_ylim(0, max(1.05, amp_max + 0.08))

        ax_v.fill_between(tr, 0, vel, where=(vel >= 0), color=_COL_OPEN,
                          alpha=0.45, interpolate=True, label="Opening")
        ax_v.fill_between(tr, 0, vel, where=(vel < 0), color=_COL_CLOS,
                          alpha=0.45, interpolate=True, label="Closing")
        ax_v.plot(tr, vel, color="#ce93d8", linewidth=0.8, alpha=0.8)
        ax_v.axhline(0, color=_TEXT, linewidth=0.5, linestyle="--", alpha=0.3)
        self._style_ax(ax_v, "Velocity  (ratio/s)", "Vel.")
        vel_lim = _sym_limit(vel)
        ax_v.set_ylim(-vel_lim, vel_lim)

        ax_s.fill_between(tr, 0, spd, color=_COL_SPD, alpha=0.45,
                          interpolate=True, label="Speed")
        ax_s.plot(tr, spd, color="#e040fb", linewidth=0.8, alpha=0.9)
        self._style_ax(ax_s, "Speed  (ratio/s)", "Spd.")
        ax_s.set_ylim(0, _sym_limit(spd))

        x0, x1 = self._target_xlim(self._cursor_time()) or (t0, t1)
        self._view_x0 = x0
        for ax in self._axes:
            ax.set_xlim(x0, x1)
            self._legend(ax)

        self._cursors = [ax.axvline(x=self._cursor_time(), color=_COL_CUR, linewidth=2.0,
                                    zorder=12, alpha=0.9, animated=True)
                         for ax in self._axes]
        self._wave_canvas.draw()

    def _on_wave_draw(self, _event):
        self._wave_bg = self._wave_canvas.copy_from_bbox(self._fig.bbox)
        for ax, ln in zip(self._axes, self._cursors):
            ax.draw_artist(ln)

    def _on_wave_resize(self, _event):
        self._wave_bg = None

    def _cursor_time(self):
        if self.cap is not None:
            return self._signal_time()
        return float(self.sig_t[0]) if self.sig_t is not None else 0.0

    def _target_xlim(self, t):
        if self.sig_t is None:
            return None
        t0, t1 = float(self.sig_t[0]), float(self.sig_t[-1])
        w = self._view_width
        if w is None or w >= t1 - t0:
            return t0, t1
        x0 = t - w / 2 if self._follow_var.get() else self._view_x0
        x0 = min(max(x0, t0), t1 - w)
        return x0, x0 + w

    def _scroll_view(self, t, force=False):
        lim = self._target_xlim(t)
        if lim is None or not self._cursors:
            return False
        cur = self._axes[0].get_xlim()
        if abs(cur[0] - lim[0]) < 1e-9 and abs(cur[1] - lim[1]) < 1e-9:
            return False
        now = time.perf_counter()
        if not force and now - self._last_scroll < 0.05:
            return False
        self._last_scroll = now
        for ax in self._axes:
            ax.set_xlim(*lim)
        self._view_x0 = lim[0]
        self._wave_bg = None
        self._wave_canvas.draw_idle()
        return True

    def _set_view_label(self):
        w = self._view_width
        self._view_var.set("Full" if w is None else f"{w:g} s")

    def _on_view_selected(self, _event=None):
        text = self._view_var.get()
        self._view_width = None if text == "Full" else float(text.split()[0])
        self.root.focus_set()
        self._refresh_view()

    def _on_follow_toggle(self):
        self._view_x0 = self._axes[0].get_xlim()[0]
        self._refresh_view()

    def _refresh_view(self):
        if self.sig_t is None:
            return
        t = self._cursor_time()
        self._scroll_view(t, force=True)
        self._update_cursor(t)

    def _on_wave_scroll(self, event):
        if self.sig_t is None or event.inaxes not in self._axes:
            return
        span = float(self.sig_t[-1] - self.sig_t[0])
        x0, x1 = self._axes[0].get_xlim()
        width = x1 - x0
        if event.key == "shift" or "shift" in (getattr(event, "modifiers", None) or ()):
            self._follow_var.set(False)
            self._view_width = width if width < span else None
            self._view_x0 = x0 - event.step * 0.2 * width
        else:
            new_width = float(np.clip(width * (0.8 ** event.step), 1.0, span))
            self._view_width = None if new_width >= span else round(new_width, 2)
            centre = self._cursor_time() if self._follow_var.get() else event.xdata
            self._view_x0 = centre - new_width / 2
        self._set_view_label()
        self._refresh_view()

    def _update_cursor(self, t_rel):
        if not self._cursors:
            return
        for ln in self._cursors:
            ln.set_xdata([t_rel, t_rel])
        if self._scroll_view(t_rel):
            return
        if self._wave_bg is None:
            self._wave_canvas.draw_idle()
            return
        self._wave_canvas.restore_region(self._wave_bg)
        for ax, ln in zip(self._axes, self._cursors):
            ax.draw_artist(ln)
        self._wave_canvas.blit(self._fig.bbox)

    def _on_wave_click(self, event):
        if (event.button != 1 or event.inaxes not in self._axes or event.xdata is None
                or self.cap is None or self.sig_t is None):
            return
        self._seek_frame(int(round((event.xdata + self._t_offset) * self.src_fps)))

    def _signal_time(self):
        return self.current_fidx / self.src_fps - self._t_offset

    def _set_activity_label(self, text):
        if text != self._activity_label:
            self._activity_label = text
            self._activity_status.configure(text=text)

    def _switch_hand(self):
        hand = self._hand_var.get()
        if self._hand_data.get(hand) is None:
            if self._shown_hand is not None:
                self._hand_var.set(self._shown_hand)
            messagebox.showinfo("No data", "Run the pipeline first.")
            return
        self._auto_hand.set(False)
        self._set_activity_label("Manual display")
        self._load_hand_signals(hand)
        self._draw_waveforms()
        if self.cap is not None:
            self._update_cursor(self._signal_time())

    def _auto_changed(self):
        if not self._auto_hand.get():
            self._set_activity_label("Manual display")
            return
        if self._activity is None:
            self._set_activity_label(_NO_ACTIVITY)
            return
        self._apply_auto_hand(self._signal_time())
        if self.cap is not None:
            self._update_cursor(self._signal_time())

    def _apply_auto_hand(self, seconds):
        if not self._auto_hand.get() or self._activity is None:
            return
        hand, label = self._activity.at(seconds)
        self._set_activity_label("Auto: " + label)
        if (hand is not None and self._hand_data.get(hand) is not None
                and self._shown_hand != hand):
            self._hand_var.set(hand)
            self._load_hand_signals(hand)
            self._draw_waveforms()

    def _load_hand_signals(self, hand):
        d = self._hand_data.get(hand)
        if d is None:
            return
        self._shown_hand = hand
        self.sig_t = d["t"]
        self.sig_amp = d["amp"]
        self.sig_vel = d["vel"]
        self.sig_spd = d["spd"]
        self.sig_tap = d["tap_mask"]

    def _reset_analysis(self):
        self._session += 1
        self._activity = None
        self._hand_data = {"Right": None, "Left": None}
        self._shown_hand = None
        self._analysis_window = None
        self.sig_t = self.sig_amp = self.sig_vel = self.sig_spd = self.sig_tap = None
        self._draw_placeholders()
        self._set_activity_label(_NO_ACTIVITY if self._auto_hand.get() else "Manual display")
        self._right_badge.configure(text="Right: not computed", fg=_MUTED)
        self._left_badge.configure(text="Left: not computed", fg=_MUTED)

    def _open_video(self):
        if self._extracting:
            messagebox.showinfo("Extraction running", "Cancel or wait before opening another video.")
            return
        path = filedialog.askopenfilename(
            title="Open Hand Recording",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All", "*.*")]
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0:
            cap.release()
            messagebox.showerror("Error", "Cannot determine the number of frames in this video.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
            messagebox.showwarning("Unknown FPS", "Video reports no frame rate; assuming 30 fps.")

        self._stop_playback()
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self._cap_pos = 0
        self._last_raw = None
        self._last_raw_idx = -1
        self.src_fps = float(fps)
        self.src_frames = frames
        self.src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_fidx = 0
        self.zoom_rect = None

        self._video_path = path
        self._overlay = []
        self._xlsx_path = ""
        self._xlsx_lbl.configure(text="No file", fg=_MUTED)
        self._reset_analysis()
        self._pipe_status.configure(text="Ready", fg=_MUTED)
        self._mp_status.configure(text="Video loaded. Extract landmarks or load an XLSX.")
        self._fps_var.set(f"{self.src_fps:.6g}")

        self._tl.configure(to=max(1, frames - 1))
        self._suppress_seek = True
        self._tl_var.set(0)
        self._suppress_seek = False
        self._vid_status.configure(
            text=f"{os.path.basename(path)}  {self.src_w}×{self.src_h}  "
                 f"{self.src_fps:.2f} fps  {frames / self.src_fps:.1f} s"
        )
        if self._placeholder_id:
            self.vid_canvas.delete(self._placeholder_id)
            self._placeholder_id = None
        if not self._show_frame(0):
            messagebox.showerror("Error", "Could not read the first frame of this video.")

    def _collect_ui_cycles(self):
        if self._closing:
            return
        gc.collect()
        self.root.after(1000, self._collect_ui_cycles)

    def _callback_error(self, exc_type, value, tb):
        details = "".join(traceback.format_exception(exc_type, value, tb))
        print(details)
        if not self._closing:
            messagebox.showerror("GUI error", details, parent=self.root)

    def _extract_landmarks(self):
        if self._extracting:
            return
        if not self._video_path:
            messagebox.showinfo("No video", "Open a video first.")
            return
        model = _find_model()
        if model is None:
            messagebox.showerror("Model missing",
                                 f"Place hand_landmarker.task in:\n{_app_dir()}",
                                 parent=self.root)
            return
        self._stop_playback()
        try:
            output_dir = _output_dir()
        except OSError as exc:
            messagebox.showerror("Cannot save", str(exc), parent=self.root)
            return
        stem = os.path.splitext(os.path.basename(self._video_path))[0]
        output = os.path.join(
            output_dir, stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_landmarks.xlsx")

        if self._extract_queue is not None:
            self._extract_queue.close()
            self._extract_queue.cancel_join_thread()
        self._extract_queue = self._mp_context.Queue()
        self._extracting = True
        self._exit_seen = False
        self._cancel_extract.clear()
        self._mp_btn.configure(state=tk.DISABLED)
        self._cancel_btn.configure(state=tk.NORMAL)
        self._mp_status.configure(text="Starting MediaPipe …")
        self._extract_process = self._mp_context.Process(
            target=extraction_worker,
            args=(self._video_path, model, output, self._swap_mp.get(),
                  self._extract_queue, self._cancel_extract),
            daemon=True)
        self.root.after(50, self._start_extraction_process)

    def _start_extraction_process(self):
        if self._closing:
            return
        try:
            self._extract_process.start()
        except Exception:
            self._finish_extraction()
            self._mp_status.configure(text="Could not start extraction")
            messagebox.showerror("Extraction error", traceback.format_exc(), parent=self.root)
            return
        self.root.after(100, self._poll_extraction)

    def _cancel_extraction(self):
        if self._extracting:
            self._cancel_extract.set()
            self._mp_status.configure(text="Cancelling …")

    def _finish_extraction(self):
        self._extracting = False
        self._mp_btn.configure(state=tk.NORMAL)
        self._cancel_btn.configure(state=tk.DISABLED)
        self.root.after(1000, multiprocessing.active_children)

    def _poll_extraction(self):
        if self._closing or not self._extracting:
            return
        try:
            while True:
                kind, payload = self._extract_queue.get_nowait()
                if kind == "progress":
                    n, total = payload
                    if total:
                        self._mp_status.configure(
                            text=f"Processing {n} / {total} frames  ({100 * n / total:.0f}%)")
                    else:
                        self._mp_status.configure(text=f"Processing {n} frames")
                    continue
                self._finish_extraction()
                if kind == "done":
                    path, fps, overlay = payload
                    self._overlay = overlay
                    self._xlsx_path = path
                    self._reset_analysis()
                    self._fps_var.set(f"{fps:.6g}")
                    self._xlsx_lbl.configure(text=os.path.basename(path), fg=_COL_TAP)
                    self._mp_status.configure(text="Extracted. Run Pipeline to analyse the full video.")
                    self._pipe_status.configure(text="Ready", fg=_MUTED)
                    self._render()
                elif kind == "cancelled":
                    self._mp_status.configure(text="Extraction cancelled")
                else:
                    self._mp_status.configure(text="Extraction failed")
                    messagebox.showerror("MediaPipe error", payload, parent=self.root)
                return
        except queue.Empty:
            pass
        proc = self._extract_process
        if proc is not None and proc.exitcode is not None:
            if not self._exit_seen:
                self._exit_seen = True
                self.root.after(300, self._poll_extraction)
                return
            self._finish_extraction()
            self._mp_status.configure(text="Extraction process exited unexpectedly")
            messagebox.showerror("Extraction process stopped",
                                 f"Exit code: {proc.exitcode}. Check terminal output.",
                                 parent=self.root)
            return
        self.root.after(100, self._poll_extraction)

    def _open_xlsx(self):
        if self._extracting:
            messagebox.showinfo("Extraction running", "Wait for landmark extraction to finish.")
            return
        path = filedialog.askopenfilename(
            title="Load Landmark XLSX",
            filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")]
        )
        if not path:
            return
        self._xlsx_path = path
        self._overlay = []
        self._reset_analysis()
        self._render()
        fps = _xlsx_fps(path)
        if fps is not None:
            self._fps_var.set(f"{fps:.6g}")
        self._xlsx_lbl.configure(text=os.path.basename(path), fg=_COL_TAP)
        self._pipe_status.configure(text="XLSX loaded → Run Pipeline (full recording)", fg=_MUTED)

    def _parse_window(self):
        def value(var, name):
            text = var.get().strip()
            if not text:
                return None
            v = float(text)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"Analysis {name} must be a time in seconds ≥ 0.")
            return v
        start = value(self._win_start_var, "start")
        end = value(self._win_end_var, "end")
        if start is not None and end is not None and end <= start + 1.0:
            raise ValueError("Analysis end must be at least 1 s after the start.")
        return start, end

    def _set_window_edge(self, var):
        if self.cap is None:
            messagebox.showinfo("No video", "Open a video and move to the frame you want first.")
            return
        var.set(f"{max(0.0, self._signal_time()):.2f}")
        self.root.focus_set()

    def _clear_window(self):
        self._win_start_var.set("")
        self._win_end_var.set("")
        self.root.focus_set()

    def _run_pipeline(self):
        if self._pipeline_busy:
            return
        self.root.focus_set()
        if self._extracting:
            messagebox.showinfo("Extraction running", "Wait for landmark extraction to finish.")
            return
        if not _PIPELINE_OK:
            messagebox.showerror("Pipeline unavailable", _PIPELINE_ERR)
            return
        if not self._xlsx_path:
            messagebox.showwarning("No XLSX", "Load an XLSX file or extract landmarks first.")
            return
        try:
            fps = float(self._fps_var.get())
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("FPS must be a positive number.")
            t_start, t_end = self._parse_window()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc) or "FPS must be a positive number.")
            return

        self._pipeline_busy = True
        self._pipe_btn.configure(state=tk.DISABLED)
        self._pipe_status.configure(text="⏳  Running pipeline …", fg=_COL_TAP)
        self._right_badge.configure(text="Right: computing …", fg=_COL_TAP)
        self._left_badge.configure(text="Left:  computing …", fg=_COL_TAP)
        out = queue.Queue()
        threading.Thread(target=_pipeline_job,
                         args=(self._xlsx_path, fps, t_start, t_end, out),
                         daemon=True).start()
        self.root.after(100, self._poll_pipeline, out, self._session)

    def _poll_pipeline(self, out, session):
        if self._closing:
            return
        try:
            kind, payload = out.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_pipeline, out, session)
            return
        self._pipeline_busy = False
        self._pipe_btn.configure(state=tk.NORMAL)
        if session != self._session:
            return
        if kind == "error":
            self._right_badge.configure(text="Right: not computed", fg=_MUTED)
            self._left_badge.configure(text="Left: not computed", fg=_MUTED)
            self._pipe_status.configure(text="Error — see dialog", fg=_ERR_RED)
            messagebox.showerror("Pipeline error", payload, parent=self.root)
            return
        self._apply_pipeline(*payload)

    def _apply_pipeline(self, activity, hands, window):
        self._activity = activity
        self._analysis_window = window
        summary_parts = []
        for hand in ("Right", "Left"):
            badge = self._right_badge if hand == "Right" else self._left_badge
            data = hands[hand]
            self._hand_data[hand] = data
            hf = data["hf"]
            cov = 100 * data["coverage"]
            if hf is None:
                badge.configure(text=f"{hand}: no taps — {data['reason']}  ·  tracked {cov:.0f}%",
                                fg=_WARN)
                summary_parts.append(f"{hand}: no taps")
                continue
            n_taps = int(data["tap_mask"].sum())
            cd = hf["cycle_data"]["mean_cycle_dur"]
            amp_pct = hf["hb_features"]["amp_avg_pct"]
            cv = hf["cycle_data"]["cv_iti"]
            badge.configure(
                text=(f"{hand}: ✓  {n_taps} taps  tracked {cov:.0f}%  "
                      f"Amp={amp_pct:.1f}%  CD={cd:.3f}s  CV={cv:.3f}"),
                fg=_OK_GREEN)
            summary_parts.append(f"{hand}: {n_taps} taps  Amp={amp_pct:.1f}%  CV={cv:.3f}")

        current = self._hand_var.get()
        with_taps = [h for h in (current, "Right", "Left") if hands[h]["hf"] is not None]
        chosen = with_taps[0] if with_taps else current
        self._hand_var.set(chosen)
        self._load_hand_signals(chosen)
        self._draw_waveforms()
        self._auto_changed()
        if self.cap is not None:
            self._update_cursor(self._signal_time())
        w0, w1 = window
        status_fg = _OK_GREEN if with_taps else _WARN
        self._pipe_status.configure(
            text=f"✓ {w0:.1f}–{w1:.1f}s | " + "   |   ".join(summary_parts), fg=status_fg)

    def _on_canvas_resize(self, e):
        if e.width < 2 or e.height < 2:
            return
        self._cw, self._ch = e.width, e.height
        if self._placeholder_id:
            self.vid_canvas.coords(self._placeholder_id, e.width / 2, e.height / 2)
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(40, self._after_resize)

    def _after_resize(self):
        self._resize_job = None
        if not self.playing:
            self._render()

    def _view_geometry(self):
        cw, ch = self._cw, self._ch
        if self.zoom_rect:
            zx1, zy1, zx2, zy2 = self.zoom_rect
            w, h, x0, y0 = zx2 - zx1, zy2 - zy1, zx1, zy1
        else:
            w, h, x0, y0 = self.src_w, self.src_h, 0, 0
        sc = min(cw / w, ch / h) if w and h else 1.0
        ox = (cw - int(w * sc)) // 2
        oy = (ch - int(h * sc)) // 2
        return x0, y0, sc, ox, oy

    def _disp_to_src(self, dx, dy):
        x0, y0, sc, ox, oy = self._view_geometry()
        sx = x0 + (dx - ox) / sc
        sy = y0 + (dy - oy) / sc
        return (float(np.clip(sx, 0, self.src_w - 1)),
                float(np.clip(sy, 0, self.src_h - 1)))

    def _on_press(self, e):
        self.root.focus_set()
        self._drag_start = (e.x, e.y)
        if self._rect_id:
            self.vid_canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        if self._rect_id:
            self.vid_canvas.coords(self._rect_id, x0, y0, e.x, e.y)
        else:
            self._rect_id = self.vid_canvas.create_rectangle(
                x0, y0, e.x, e.y, outline=_COL_TAP, width=2, dash=(5, 3))

    def _on_release(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = e.x, e.y
        self._drag_start = None
        if self._rect_id:
            self.vid_canvas.delete(self._rect_id)
            self._rect_id = None
        if self.cap is None or abs(x1 - x0) < 12 or abs(y1 - y0) < 12:
            return
        sx0, sy0 = self._disp_to_src(min(x0, x1), min(y0, y1))
        sx1, sy1 = self._disp_to_src(max(x0, x1), max(y0, y1))
        if (sx1 - sx0) < 20 or (sy1 - sy0) < 20:
            return
        self.zoom_rect = (int(sx0), int(sy0), int(sx1), int(sy1))
        self._render()

    def _reset_zoom(self):
        self.zoom_rect = None
        self._render()

    def _read_frame(self, fidx):
        cap = self.cap
        if cap is None:
            return None
        if fidx == self._last_raw_idx and self._last_raw is not None:
            return self._last_raw
        pos = self._cap_pos
        if pos is None or fidx < pos or fidx - pos > 12:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        else:
            for _ in range(fidx - pos):
                if not cap.grab():
                    self._cap_pos = None
                    return None
        ok, frame = cap.read()
        self._cap_pos = fidx + 1 if ok else None
        return frame if ok else None

    def _show_frame(self, fidx):
        if self.cap is None or self.src_frames <= 0:
            return False
        fidx = int(np.clip(fidx, 0, self.src_frames - 1))
        raw = self._read_frame(fidx)
        if raw is None:
            return False
        self._last_raw = raw
        self._last_raw_idx = fidx
        self.current_fidx = fidx
        self._render()
        self._sync_time()
        return True

    def _sync_time(self):
        t_abs = self.current_fidx / self.src_fps
        self._time_lbl.configure(
            text=f"{_fmt(t_abs)} / {_fmt(self.src_frames / self.src_fps)}")
        self._suppress_seek = True
        self._tl_var.set(self.current_fidx)
        self._suppress_seek = False
        t_sig = t_abs - self._t_offset
        self._apply_auto_hand(t_sig)
        if self.sig_t is not None:
            self._update_cursor(t_sig)

    def _render(self):
        raw = self._last_raw
        if raw is None:
            return
        fidx = self._last_raw_idx
        if 0 <= fidx < len(self._overlay) and self._overlay[fidx]:
            raw = draw_overlay(raw, self._overlay[fidx])

        disp = raw
        if self.zoom_rect:
            x1, y1, x2, y2 = self.zoom_rect
            x1 = max(0, min(x1, self.src_w - 1))
            x2 = max(x1 + 2, min(x2, self.src_w))
            y1 = max(0, min(y1, self.src_h - 1))
            y2 = max(y1 + 2, min(y2, self.src_h))
            disp = raw[y1:y2, x1:x2]
            self._zoom_lbl.configure(
                text=f"Zoom  {x1},{y1}→{x2},{y2}  ({x2 - x1}×{y2 - y1} px)")
        else:
            self._zoom_lbl.configure(text="Zoom: full frame")

        cw, ch = self._cw, self._ch
        h, w = disp.shape[:2]
        sc = min(cw / w, ch / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR
        sized = cv2.resize(disp, (nw, nh), interpolation=interp)
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = sized

        mw, mh = 130, 98
        if self.zoom_rect and cw > mw + 8 and ch > mh + 8:
            mini = cv2.resize(self._last_raw, (mw, mh), interpolation=cv2.INTER_AREA)
            zx1, zy1, zx2, zy2 = self.zoom_rect
            cv2.rectangle(mini,
                          (int(zx1 / self.src_w * mw), int(zy1 / self.src_h * mh)),
                          (int(zx2 / self.src_w * mw), int(zy2 / self.src_h * mh)),
                          (79, 213, 255), 1)
            cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), (200, 100, 80), 1)
            cv2.putText(mini, "overview", (3, mh - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, (210, 160, 160), 1, cv2.LINE_AA)
            canvas[4:4 + mh, cw - mw - 4:cw - 4] = mini

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.vid_canvas.itemconfigure(self._img_id, image=self._photo)

    def _toggle_play(self):
        if self.cap is None:
            return
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _set_speed(self, _=None):
        self.play_speed = float(self._speed_var.get())
        self._spd_lbl.configure(text=f"{self.play_speed:.2f}×")
        if self.playing:
            self._anchor_playback()

    def _anchor_playback(self):
        self._play_t0 = time.perf_counter()
        self._play_f0 = self.current_fidx

    def _frame_rate(self):
        return self.src_fps * max(0.05, self.play_speed)

    def _start_playback(self):
        if self.cap is None or self.src_frames <= 0 or self._closing or self.playing:
            return
        if self.current_fidx >= self.src_frames - 1:
            self._show_frame(0)
        self.playing = True
        self._play_btn.configure(text="⏸  Pause")
        self._anchor_playback()
        self._schedule_tick(1)

    def _replay(self):
        if self.cap is None or self.src_frames <= 0:
            return
        self._stop_playback()
        self._show_frame(0)
        self._start_playback()

    def _schedule_tick(self, delay_ms):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(max(1, int(delay_ms)), self._tick)

    def _stop_playback(self):
        self.playing = False
        self._play_btn.configure(text="▶  Play")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _wrap_or_stop(self):
        if self._loop_var.get():
            self._show_frame(0)
            self._anchor_playback()
            return True
        self._stop_playback()
        return False

    def _tick(self):
        self._after_id = None
        if not self.playing or self._closing or self.cap is None:
            return
        rate = self._frame_rate()
        target = self._play_f0 + int((time.perf_counter() - self._play_t0) * rate)
        last = self.src_frames - 1
        if target > last:
            if not self._loop_var.get():
                self._show_frame(last)
            if not self._wrap_or_stop():
                return
        elif target > self.current_fidx and not self._show_frame(target):
            if not self._wrap_or_stop():
                return
        due = self._play_t0 + (self.current_fidx - self._play_f0 + 1) / rate
        self._schedule_tick((due - time.perf_counter()) * 1000)

    def _step(self, d):
        if self.cap is None:
            return
        self._stop_playback()
        self._show_frame(self.current_fidx + d)

    def _seek_frame(self, fidx):
        if self.cap is None:
            return
        self._show_frame(fidx)
        if self.playing:
            self._anchor_playback()

    def _on_seek(self, value=None):
        if self.cap is None or self._suppress_seek:
            return
        fidx = int(round(float(self._tl_var.get() if value is None else value)))
        if fidx != self.current_fidx:
            self._seek_frame(fidx)

    def _apply_offset(self):
        try:
            offset = float(self._offset_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Offset must be a number (seconds).")
            return
        self._t_offset = offset
        self.root.focus_set()
        if self.cap is not None:
            self._sync_time()

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
        for timer in self.root.tk.call("after", "info"):
            self.root.after_cancel(timer)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._photo = None
        gc.collect()
        self.root.destroy()
        if self._extract_queue is not None:
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