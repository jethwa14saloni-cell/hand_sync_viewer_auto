import numpy as np
import pandas as pd
import warnings
from hand_activity import ActivityTimeline
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Keep numerical warnings visible for data quality review.


class TapExtractor:
    def __init__(self, excel_path, fps=59.97):
        self.excel_path = excel_path
        self.fps = fps
        self.df          = None
        self.time        = None
        self.right_dist_normalized = None
        self.left_dist_normalized  = None
        self.right_hand_time       = None
        self.left_hand_time        = None
        self.right_dist_raw        = None
        self.left_dist_raw         = None

    def load_excel_dataset(self):
        self.df   = pd.read_excel(self.excel_path)
        self.time = self.df["time_seconds"].to_numpy(dtype=float)
        if len(self.time) < 3 or not np.all(np.isfinite(self.time)) or np.any(np.diff(self.time) <= 0):
            raise ValueError("Need at least 3 rows with finite, strictly increasing time_seconds.")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("FPS must be positive.")
    def _butter_lowpass_filter(self, signal, cutoff=11.0, order=4):
        nyquist = self.fps / 2.0
        cutoff = min(cutoff, nyquist * 0.8)
        b, a    = butter(order, cutoff / nyquist, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def _dist3(self, ax, ay, az, bx, by, bz):
        return np.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)

    def compute_distances(self):
        df = self.df

        right_d1 = self._dist3(
            df["right_hand_index_tip_x"].values, df["right_hand_index_tip_y"].values,
            df["right_hand_index_tip_z"].values,
            df["right_hand_thumb_tip_x"].values, df["right_hand_thumb_tip_y"].values,
            df["right_hand_thumb_tip_z"].values,
        )
        right_d2 = self._dist3(
            df["right_hand_wrist_x"].values, df["right_hand_wrist_y"].values, df["right_hand_wrist_z"].values,
            df["right_hand_index_mcp_x"].values, df["right_hand_index_mcp_y"].values,
            df["right_hand_index_mcp_z"].values,
        )

        left_d1 = self._dist3(
            df["left_hand_index_tip_x"].values, df["left_hand_index_tip_y"].values, df["left_hand_index_tip_z"].values,
            df["left_hand_thumb_tip_x"].values, df["left_hand_thumb_tip_y"].values, df["left_hand_thumb_tip_z"].values,
        )
        left_d2 = self._dist3(
            df["left_hand_wrist_x"].values, df["left_hand_wrist_y"].values, df["left_hand_wrist_z"].values,
            df["left_hand_index_mcp_x"].values, df["left_hand_index_mcp_y"].values, df["left_hand_index_mcp_z"].values,
        )

        right_d2 = np.where(right_d2 < 1e-6, np.nan, right_d2)
        left_d2 = np.where(left_d2 < 1e-6, np.nan, left_d2)

        right_norm_ref = float(np.nanmedian(right_d2)) if np.any(np.isfinite(right_d2)) else np.nan
        left_norm_ref = float(np.nanmedian(left_d2)) if np.any(np.isfinite(left_d2)) else np.nan
        valid_refs = [v for v in (right_norm_ref,left_norm_ref) if np.isfinite(v) and v > 0]
        if not valid_refs:
            raise ValueError("No valid palm reference in either hand.")
        shared_ref = float(np.mean(valid_refs))

        self.right_dist_raw = right_d1
        self.left_dist_raw = left_d1

        right_normalized_full = np.clip(right_d1 / shared_ref, 0, None)
        left_normalized_full = np.clip(left_d1 / shared_ref, 0, None)

        right_filtered_full = self._filter_valid_segments(right_normalized_full)
        left_filtered_full = self._filter_valid_segments(left_normalized_full)

        self.right_hand_time = self.time.copy()
        self.left_hand_time = self.time.copy()
        self.right_dist_normalized = right_filtered_full
        self.left_dist_normalized = left_filtered_full
        print(f"  Full recording: {self.time[0]:.3f}–{self.time[-1]:.3f}s; shared palm reference = {shared_ref:.4f}")

    def _filter_valid_segments(self, signal):
        signal = np.asarray(signal, dtype=float).copy()
        signal[~np.isfinite(signal)] = np.nan
        # Fill only internal gaps <= 100 ms; never fill leading/trailing gaps.
        limit = max(1, int(self.fps * 0.1))
        valid = np.flatnonzero(np.isfinite(signal))
        for a,b in zip(valid[:-1],valid[1:]):
            if 1 < b-a <= limit+1:
                signal[a:b+1] = np.linspace(signal[a],signal[b],b-a+1)
        finite = np.isfinite(signal)
        changes = np.diff(np.r_[False,finite,False].astype(int))
        for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            if end-start > 15:
                signal[start:end] = self._butter_lowpass_filter(signal[start:end])
            else:
                signal[start:end] = np.nan
        return signal

    def _get_candidate_peaks(self, signal, min_tap_interval_ms=250,
                             prominence_ratio=0.35, min_height_ratio=0.45):
        smoothed = gaussian_filter1d(signal, sigma=2)
        sig_range = float(np.max(smoothed) - np.min(smoothed))
        min_distance = int(min_tap_interval_ms / 1000.0 * self.fps)

        valleys, _ = find_peaks(
            -smoothed,
            distance=min_distance,
            prominence=prominence_ratio * sig_range * 0.5,
        )

        if len(valleys) < 2:
            peaks, _ = find_peaks(
                smoothed,
                prominence=prominence_ratio * sig_range,
                distance=min_distance,
                height=np.min(smoothed) + min_height_ratio * sig_range,
            )
            return np.array(peaks), sig_range

        refined = []
        for i in range(len(valleys) - 1):
            lo = valleys[i]
            hi = valleys[i + 1]
            seg = signal[lo: hi + 1]
            peak_idx = lo + int(np.argmax(seg))
            refined.append(peak_idx)

        peaks = np.array(refined)

        if len(peaks) > 1:
            merged = [peaks[0]]
            for pk in peaks[1:]:
                if pk - merged[-1] < min_distance:
                    if signal[pk] > signal[merged[-1]]:
                        merged[-1] = pk
                else:
                    merged.append(pk)
            peaks = np.array(merged)


        return peaks, sig_range

    def _get_reference_stats(self, signal, peaks):
        s      = len(peaks) // 4
        e      = 3 * len(peaks) // 4
        stable = peaks[s:e]

        avg_peak   = np.mean(signal[stable])
        std_peak   = np.std(signal[stable])
        valleys    = [np.min(signal[stable[i]:stable[i+1]]) for i in range(len(stable)-1)]
        avg_valley = np.mean(valleys)
        std_valley = np.std(valleys)

        return avg_peak, std_peak, avg_valley, std_valley, avg_peak - avg_valley

    def tap_detection(self):
        signals = {
            "Right": (self.right_dist_normalized, self.right_hand_time),
            "Left" : (self.left_dist_normalized,  self.left_hand_time),
        }
        results = {}
        self.activity = ActivityTimeline({hand: (time, signal) for hand, (signal, time) in signals.items()})

        for hand_name, (signal, time_array) in signals.items():
            signal    = np.asarray(signal, dtype=float)
            tap_start = 0
            tap_end   = len(signal) - 1

            if len(signal) < 3 or not np.all(np.isfinite(signal)):
                print(f"  [{hand_name}] skipped: missing data / long detection gap in full recording.")
                results[hand_name] = {"peaks": np.array([], dtype=int),
                    "tap_start_frame": tap_start, "tap_end_frame": tap_end}
                continue

            if float(np.max(signal) - np.min(signal)) < 0.05:
                print(f"  [{hand_name}] signal range too small — skipping")
                results[hand_name] = {"peaks": np.array([]),
                                      "tap_start_frame": tap_start,
                                      "tap_end_frame":   tap_end}
                continue

            peaks, _ = self._get_candidate_peaks(signal)

            if len(peaks) < 4:
                print(f"  [{hand_name}] fewer than 4 peaks found — skipping")
                results[hand_name] = {"peaks": np.array([]),
                                      "tap_start_frame": tap_start,
                                      "tap_end_frame":   tap_end}
                continue
            peaks = peaks[1:-1]
            peaks = np.array([pk for pk in peaks
                if self.activity.supported(hand_name, time_array[int(pk)])], dtype=int)
            if len(peaks) < 3:
                print(f"  [{hand_name}] no sustained tapping activity — skipping")
                results[hand_name] = {"peaks": np.array([], dtype=int),
                    "tap_start_frame": tap_start, "tap_end_frame": tap_end}
                continue


            if len(peaks) < 2:
                print(f"  [{hand_name}] not enough taps after trimming — skipping")
                results[hand_name] = {"peaks": np.array([]),
                                      "tap_start_frame": tap_start,
                                      "tap_end_frame":   tap_end}
                continue

            avg_peak, std_peak, avg_valley, std_valley, avg_amp = \
                self._get_reference_stats(signal, peaks)

            print(f"  [{hand_name}] {len(peaks)} taps  |  "
                  f"avg amplitude = {avg_amp:.4f}  |  "
                  f"avg peak = {avg_peak:.4f}  |  "
                  f"avg valley = {avg_valley:.4f}")

            results[hand_name] = {"peaks"          : peaks,
                                  "tap_start_frame": tap_start,
                                  "tap_end_frame"  : tap_end}
        return results

    def run(self):
        self.load_excel_dataset()
        self.compute_distances()
        return self.tap_detection()