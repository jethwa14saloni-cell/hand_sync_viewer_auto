import numpy as np
import pandas as pd
from hand_activity import ActivityTimeline
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d


MIN_SEGMENT_SECONDS = 1.0
MAX_GAP_FILL_SECONDS = 0.25
LOCAL_RANGE_SECONDS = 1.0
MIN_VALLEY_DEPTH = 0.06
GLOBAL_DEPTH_RATIO = 0.10
HOLD_FACTOR = 2.0
HOLD_MIN_SECONDS = 1.2


def _finite_runs(signal):
    finite = np.isfinite(signal)
    changes = np.diff(np.r_[False, finite, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


class TapExtractor:
    def __init__(self, excel_path, fps=59.97, t_start=None, t_end=None):
        self.excel_path = excel_path
        self.fps = fps
        self.t_start = t_start
        self.t_end = t_end
        self.df = None
        self.time = None
        self.window_mask = None
        self.right_dist_normalized = None
        self.left_dist_normalized = None
        self.right_hand_time = None
        self.left_hand_time = None
        self.right_dist_raw = None
        self.left_dist_raw = None
        self.activity = None

    def load_excel_dataset(self):
        self.df = pd.read_excel(self.excel_path)
        self.time = self.df["time_seconds"].to_numpy(dtype=float)
        if len(self.time) < 3 or not np.all(np.isfinite(self.time)) or np.any(np.diff(self.time) <= 0):
            raise ValueError("Need at least 3 rows with finite, strictly increasing time_seconds.")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("FPS must be positive.")
        lo = -np.inf if self.t_start is None else float(self.t_start)
        hi = np.inf if self.t_end is None else float(self.t_end)
        if hi <= lo:
            raise ValueError("Analysis end must be after analysis start.")
        self.window_mask = (self.time >= lo) & (self.time <= hi)
        if self.window_mask.sum() < 3:
            raise ValueError("Analysis window contains fewer than 3 frames.")

    def _butter_lowpass_filter(self, signal, cutoff=11.0, order=4):
        nyquist = self.fps / 2.0
        cutoff = min(cutoff, nyquist * 0.8)
        b, a = butter(order, cutoff / nyquist, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def _dist3(self, ax, ay, az, bx, by, bz):
        return np.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)

    def _hand_distances(self, side):
        df = self.df
        c = lambda name: df[f"{side}_hand_{name}"].to_numpy(dtype=float)
        d1 = self._dist3(c("index_tip_x"), c("index_tip_y"), c("index_tip_z"),
                         c("thumb_tip_x"), c("thumb_tip_y"), c("thumb_tip_z"))
        d2 = self._dist3(c("wrist_x"), c("wrist_y"), c("wrist_z"),
                         c("index_mcp_x"), c("index_mcp_y"), c("index_mcp_z"))
        d2 = np.where(d2 < 1e-6, np.nan, d2)
        return d1, d2

    def compute_distances(self):
        right_d1, right_d2 = self._hand_distances("right")
        left_d1, left_d2 = self._hand_distances("left")
        refs = []
        for d2 in (right_d2, left_d2):
            d2 = d2[self.window_mask]
            if np.any(np.isfinite(d2)):
                v = float(np.nanmedian(d2))
                if np.isfinite(v) and v > 0:
                    refs.append(v)
        if not refs:
            raise ValueError("No valid palm reference in either hand.")
        shared_ref = float(np.mean(refs))

        self.right_dist_raw = right_d1
        self.left_dist_raw = left_d1

        right_norm = np.clip(right_d1 / shared_ref, 0, None)
        left_norm = np.clip(left_d1 / shared_ref, 0, None)
        right_norm[~self.window_mask] = np.nan
        left_norm[~self.window_mask] = np.nan

        self.right_hand_time = self.time.copy()
        self.left_hand_time = self.time.copy()
        self.right_dist_normalized = self._filter_valid_segments(right_norm)
        self.left_dist_normalized = self._filter_valid_segments(left_norm)
        w = self.time[self.window_mask]
        print(f"  Analysis window: {w[0]:.3f}–{w[-1]:.3f}s; shared palm reference = {shared_ref:.4f}")

    def _filter_valid_segments(self, signal):
        signal = np.asarray(signal, dtype=float).copy()
        signal[~np.isfinite(signal)] = np.nan
        limit = max(1, int(round(self.fps * MAX_GAP_FILL_SECONDS)))
        valid = np.flatnonzero(np.isfinite(signal))
        for a, b in zip(valid[:-1], valid[1:]):
            if 1 < b - a <= limit + 1:
                signal[a:b + 1] = np.linspace(signal[a], signal[b], b - a + 1)
        for start, end in _finite_runs(signal):
            if end - start > 15:
                signal[start:end] = self._butter_lowpass_filter(signal[start:end])
            else:
                signal[start:end] = np.nan
        return signal

    def _get_candidate_peaks(self, signal, min_tap_interval_ms=250,
                             prominence_ratio=0.35, min_height_ratio=0.45,
                             ref_min=None, ref_range=None):
        smoothed = gaussian_filter1d(signal, sigma=2)
        sig_min = float(np.min(smoothed)) if ref_min is None else ref_min
        sig_range = float(np.max(smoothed) - np.min(smoothed)) if ref_range is None else ref_range
        min_distance = max(1, int(min_tap_interval_ms / 1000.0 * self.fps))

        half = max(1, int(LOCAL_RANGE_SECONDS * self.fps))
        local_range = (maximum_filter1d(smoothed, 2 * half + 1)
                       - minimum_filter1d(smoothed, 2 * half + 1))
        robust_range = (float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))
                        if ref_range is None else ref_range)
        floor = max(MIN_VALLEY_DEPTH, GLOBAL_DEPTH_RATIO * robust_range)
        valleys, props = find_peaks(-smoothed, distance=min_distance, prominence=0.0)
        needed = np.maximum(prominence_ratio * 0.5 * local_range[valleys], floor)
        valleys = valleys[props["prominences"] >= needed]

        if len(valleys) < 2:
            peaks, _ = find_peaks(
                smoothed,
                prominence=prominence_ratio * sig_range,
                distance=min_distance,
                height=sig_min + min_height_ratio * sig_range,
            )
            return np.array(peaks, dtype=int), sig_range

        peaks = np.array([lo + int(np.argmax(signal[lo:hi + 1]))
                          for lo, hi in zip(valleys[:-1], valleys[1:])], dtype=int)

        if len(peaks) > 1:
            merged = [peaks[0]]
            for pk in peaks[1:]:
                if pk - merged[-1] < min_distance:
                    if signal[pk] > signal[merged[-1]]:
                        merged[-1] = pk
                else:
                    merged.append(pk)
            peaks = np.array(merged, dtype=int)

        return peaks, sig_range

    def _get_reference_stats(self, signal, peaks):
        s = len(peaks) // 4
        e = 3 * len(peaks) // 4
        stable = peaks[s:e]
        avg_peak = np.mean(signal[stable])
        std_peak = np.std(signal[stable])
        valleys = [np.nanmin(seg) for seg in
                   (signal[stable[i]:stable[i + 1]] for i in range(len(stable) - 1))
                   if np.any(np.isfinite(seg))]
        avg_valley = np.mean(valleys) if valleys else np.nan
        std_valley = np.std(valleys) if valleys else np.nan
        return avg_peak, std_peak, avg_valley, std_valley, avg_peak - avg_valley

    def _drop_holds(self, signal, peaks):
        if len(peaks) < 3:
            return peaks, 0
        finite = np.isfinite(signal)
        run_id = np.cumsum(np.r_[True, finite[1:] != finite[:-1]])
        spans = []
        for i, pk in enumerate(peaks):
            lo = peaks[i - 1] if i > 0 else max(0, 2 * pk - peaks[1])
            hi = peaks[i + 1] if i < len(peaks) - 1 else min(len(signal) - 1, 2 * pk - peaks[-2])
            same = run_id == run_id[pk]
            left = np.flatnonzero(same[lo:pk + 1]) + lo
            right = np.flatnonzero(same[pk:hi + 1]) + pk
            lv = left[np.argmin(signal[left])]
            rv = right[np.argmin(signal[right])]
            spans.append((self.time[rv] - self.time[lv]) / 2)
        spans = np.asarray(spans)
        limit = max(HOLD_FACTOR * float(np.median(spans)), HOLD_MIN_SECONDS / 2)
        keep = spans <= limit
        return peaks[keep], int((~keep).sum())

    def tap_detection(self):
        signals = {
            "Right": (self.right_dist_normalized, self.right_hand_time),
            "Left": (self.left_dist_normalized, self.left_hand_time),
        }
        results = {}
        self.activity = ActivityTimeline({hand: (time, signal) for hand, (signal, time) in signals.items()})
        min_len = int(MIN_SEGMENT_SECONDS * self.fps)

        for hand_name, (signal, time_array) in signals.items():
            signal = np.asarray(signal, dtype=float)
            runs = [(s, e) for s, e in _finite_runs(signal) if e - s >= min_len]
            start = runs[0][0] if runs else 0
            end = runs[-1][1] - 1 if runs else len(signal) - 1

            def skip(reason):
                print(f"  [{hand_name}] skipped: {reason}")
                results[hand_name] = {"peaks": np.array([], dtype=int),
                                      "tap_start_frame": int(start), "tap_end_frame": int(end),
                                      "segments": runs, "reason": reason}

            if not runs:
                skip(f"no tracked stretch of at least {MIN_SEGMENT_SECONDS:g} s")
                continue
            ranges = [float(np.ptp(signal[s:e])) for s, e in runs]
            if max(ranges) < 0.05:
                skip("finger distance hardly changes (no tapping)")
                continue

            tracked = np.concatenate([gaussian_filter1d(signal[s:e], sigma=2) for s, e in runs])
            shared = None if len(runs) == 1 else dict(ref_min=float(np.min(tracked)),
                                                      ref_range=float(np.ptp(tracked)))
            peaks = []
            for s, e in runs:
                cand, _ = self._get_candidate_peaks(signal[s:e], **(shared or {}))
                peaks.extend(int(pk) + s for pk in cand)
            if len(peaks) < 3:
                skip("fewer than 3 tap peaks found")
                continue

            peaks = np.array([pk for pk in peaks
                              if self.activity.supported(hand_name, time_array[pk])], dtype=int)
            if len(peaks) < 3:
                skip("no sustained tapping activity")
                continue

            peaks, holds = self._drop_holds(signal, peaks)
            if holds:
                print(f"  [{hand_name}] ignored {holds} hold(s) where the hand stayed open")
            if len(peaks) < 3:
                skip("fewer than 3 taps after removing holds")
                continue

            avg_peak, std_peak, avg_valley, std_valley, avg_amp = \
                self._get_reference_stats(signal, peaks)
            tracked = sum(e - s for s, e in runs) / max(1, int(self.window_mask.sum()))

            print(f"  [{hand_name}] {len(peaks)} taps in {len(runs)} tracked stretch(es), "
                  f"{100 * tracked:.0f}% of window  |  "
                  f"avg amplitude = {avg_amp:.4f}  |  "
                  f"avg peak = {avg_peak:.4f}  |  "
                  f"avg valley = {avg_valley:.4f}")

            results[hand_name] = {"peaks": peaks,
                                  "tap_start_frame": int(start),
                                  "tap_end_frame": int(end),
                                  "segments": runs,
                                  "reason": ""}
        return results

    def run(self):
        self.load_excel_dataset()
        self.compute_distances()
        return self.tap_detection()