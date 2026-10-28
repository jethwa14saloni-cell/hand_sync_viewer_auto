import numpy as np
import pandas as pd
from math import e as _e
from correct_tap_detection import TapExtractor


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _calc_entropy(signal: np.ndarray, base=None) -> float:
    n = len(signal)
    if n <= 1:
        return 0.0
    _, counts = np.unique(signal, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    b = _e if base is None else base
    return float(-np.sum(probs * np.log(probs) / np.log(b)))


def _calc_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def _trend_line(y: np.ndarray) -> tuple:
    x = np.arange(len(y), dtype=float)
    a, b = np.polyfit(x, y, 1)
    return x * a + b, float(a)


class FeatureExtractor:

    def compute_velocity_acceleration(self, signal, time_array):
        dt           = np.diff(time_array)
        velocity     = np.diff(signal) / dt
        acceleration = np.diff(velocity) / dt[:-1]
        velocity     = np.concatenate([[velocity[0]], velocity])
        acceleration = np.concatenate([[acceleration[0]], acceleration, [acceleration[-1]]])
        return velocity, acceleration

    def compute_speed(self, signal, time_array):
        dt    = np.diff(time_array)
        speed = np.abs(np.diff(signal) / dt)
        return np.concatenate([[speed[0]], speed])

    def compute_distance_metrics(self, signal, validated_peaks):
        peak_values = signal[validated_peaks]
        return {
            "mean_distance": float(np.mean(peak_values)),
            "max_distance" : float(np.max(peak_values)),
            "min_distance" : float(np.min(peak_values)),
            "std_distance" : float(np.std(peak_values)),
        }

    def compute_tap_features(self, signal, validated_peaks, time_array,
                             hand_name, tap_start_frame, tap_end_frame):
        velocity, acceleration = self.compute_velocity_acceleration(signal, time_array)

        amplitudes            = []
        peak_values_list      = []
        trough_values_list    = []
        opening_velocities    = []
        closing_velocities    = []
        peak_accelerations    = []
        opening_accelerations = []
        closing_accelerations = []
        tap_times             = []
        left_valley_idxs      = []
        right_valley_idxs     = []

        for i, peak_idx in enumerate(validated_peaks):
            peak_idx   = int(peak_idx)
            left_start = int(validated_peaks[i - 1]) if i > 0 else tap_start_frame
            right_end  = int(validated_peaks[i + 1]) if i < len(validated_peaks) - 1 \
                         else tap_end_frame

            left_valley_idx  = left_start + int(np.argmin(signal[left_start : peak_idx + 1]))
            right_valley_idx = peak_idx   + int(np.argmin(signal[peak_idx   : right_end + 1]))

            if left_valley_idx  == peak_idx:
                left_valley_idx  = left_start + int(np.argmin(signal[left_start : peak_idx + 1]))
            if right_valley_idx == peak_idx:
                right_valley_idx = peak_idx   + int(np.argmin(signal[peak_idx   : right_end + 1]))

            peak_val         = float(signal[peak_idx])
            left_valley_val  = float(signal[left_valley_idx])
            right_valley_val = float(signal[right_valley_idx])
            best_valley_val  = min(left_valley_val, right_valley_val)
            amplitude        = (peak_val - best_valley_val) * 100

            dt_open  = time_array[peak_idx] - time_array[left_valley_idx]
            dt_close = time_array[right_valley_idx] - time_array[peak_idx]
            v_open   = (peak_val - left_valley_val)  / dt_open  if dt_open  > 0 else np.nan
            v_close  = (peak_val - right_valley_val) / dt_close if dt_close > 0 else np.nan

            peak_accel          = float(acceleration[peak_idx])
            open_accel_segment  = acceleration[left_valley_idx : peak_idx + 1]
            close_accel_segment = acceleration[peak_idx : right_valley_idx + 1]
            mean_open_accel     = float(np.mean(open_accel_segment))  if len(open_accel_segment)  > 0 else np.nan
            mean_close_accel    = float(np.mean(close_accel_segment)) if len(close_accel_segment) > 0 else np.nan

            amplitudes.append(amplitude)
            peak_values_list.append(peak_val)
            trough_values_list.append(best_valley_val)
            opening_velocities.append(v_open)
            closing_velocities.append(v_close)
            peak_accelerations.append(peak_accel)
            opening_accelerations.append(mean_open_accel)
            closing_accelerations.append(mean_close_accel)
            tap_times.append(float(time_array[peak_idx]))
            left_valley_idxs.append(left_valley_idx)
            right_valley_idxs.append(right_valley_idx)

        return {
            "hand"                 : hand_name,
            "tap_times"            : tap_times,
            "peak_values"          : peak_values_list,
            "trough_values"        : trough_values_list,
            "amplitudes_pct"       : amplitudes,
            "opening_velocity"     : opening_velocities,
            "closing_velocity"     : closing_velocities,
            "mean_velocity"        : [(o + c) / 2 for o, c in
                                      zip(opening_velocities, closing_velocities)],
            "peak_acceleration"    : peak_accelerations,
            "opening_acceleration" : opening_accelerations,
            "closing_acceleration" : closing_accelerations,
            "left_valley_idxs"     : left_valley_idxs,
            "right_valley_idxs"    : right_valley_idxs,
            "validated_peaks"      : list(validated_peaks),
        }

    def build_tap_indices(self, validated_peaks, left_valley_idxs, right_valley_idxs):
        tap_indices = []
        for i, peak in enumerate(validated_peaks):
            lv        = int(left_valley_idxs[i])
            rv        = int(right_valley_idxs[i])
            mid_open  = int((lv + peak) / 2)
            mid_close = int((peak + rv) / 2)
            tap_indices.append(np.array(
                [lv, mid_open, peak, peak, mid_close, rv, rv], dtype=float))
        return tap_indices

    def compute_envelope_features(self, tap_features):
        amplitudes = np.array(tap_features["amplitudes_pct"])
        tap_times  = np.array(tap_features["tap_times"])
        n          = len(amplitudes)
        slope, _   = np.polyfit(tap_times, amplitudes, 1)
        early_mean = np.mean(amplitudes[:n // 3])
        late_mean  = np.mean(amplitudes[2 * n // 3:])
        decay      = late_mean - early_mean
        return {
            "amplitude_slope"      : float(slope),
            "early_mean_amplitude" : float(early_mean),
            "late_mean_amplitude"  : float(late_mean),
            "total_amplitude_decay": float(decay),
            "decay_percent"        : float(decay / early_mean * 100),
        }

    def compute_cycle_duration(self, tap_features):
        tap_times        = np.array(tap_features["tap_times"])
        cycle_durations  = np.diff(tap_times)
        cycle_numbers    = np.arange(1, len(cycle_durations) + 1)
        slope, intercept = np.polyfit(cycle_numbers, cycle_durations, 1)
        return {
            "cycle_durations": cycle_durations.tolist(),
            "cycle_numbers"  : cycle_numbers.tolist(),
            "mean_cycle_dur" : float(np.mean(cycle_durations)),
            "std_cycle_dur"  : float(np.std(cycle_durations)),
            "cv_iti"         : float(np.std(cycle_durations) / np.mean(cycle_durations)),
            "min_iti"        : float(np.min(cycle_durations)),
            "max_iti"        : float(np.max(cycle_durations)),
            "cycle_slope"    : float(slope),
            "trend"          : (cycle_numbers * slope + intercept).tolist(),
        }

    def compute_frequency(self, tap_features):
        cycle_data = self.compute_cycle_duration(tap_features)
        tap_times  = np.array(tap_features["tap_times"])
        return {
            "tap_frequency_hz"     : float(1.0 / cycle_data["mean_cycle_dur"]),
            "tap_frequency_overall": float(len(tap_times) / (tap_times[-1] - tap_times[0])),
        }

    def compute_hypokinesia_bradykinesia(self, tap_features):
        peak_values   = np.array(tap_features["peak_values"])
        trough_values = np.array(tap_features["trough_values"])
        tap_times     = np.array(tap_features["tap_times"])
        amp_i         = peak_values - trough_values
        cd_i          = np.diff(tap_times)
        return {
            "M"           : len(peak_values),
            "amp_i"       : amp_i.tolist(),
            "amp_i_pct"   : (amp_i * 100).tolist(),
            "amp_avg"     : float(np.mean(amp_i)),
            "amp_avg_pct" : float(np.mean(amp_i) * 100),
            "cd_i"        : cd_i.tolist(),
            "cd_avg"      : float(np.sum(cd_i) / (len(peak_values) - 1)),
        }

    def compute_acceleration_features(self, tap_features):
        peak_acc  = np.array(tap_features["peak_acceleration"])
        open_acc  = np.array(tap_features["opening_acceleration"])
        close_acc = np.array(tap_features["closing_acceleration"])
        tap_times = np.array(tap_features["tap_times"])
        slope, _  = np.polyfit(tap_times, peak_acc, 1)
        return {
            "mean_peak_accel" : float(np.nanmean(peak_acc)),
            "std_peak_accel"  : float(np.nanstd(peak_acc)),
            "peak_accel_slope": float(slope),
            "mean_open_accel" : float(np.nanmean(open_acc)),
            "std_open_accel"  : float(np.nanstd(open_acc)),
            "mean_close_accel": float(np.nanmean(close_acc)),
            "std_close_accel" : float(np.nanstd(close_acc)),
            "peak_acc_array"  : peak_acc.tolist(),
            "open_acc_array"  : open_acc.tolist(),
            "close_acc_array" : close_acc.tolist(),
        }

    def compute_rms_per_tap(self, signal, tap_indices, fs, to_norm=False):
        rms_vals = _nan_array(len(tap_indices))
        for n, tap in enumerate(tap_indices):
            sel1 = int(tap[0]); sel2 = int(tap[-1])
            if sel1 >= sel2 or sel2 > len(signal):
                continue
            seg = signal[sel1:sel2]
            rms_vals[n] = _calc_rms(seg)
            if to_norm and fs:
                rms_vals[n] /= (len(seg) / fs)
        return {
            "rms_per_tap": rms_vals.tolist(),
            "mean_rms"   : float(np.nanmean(rms_vals)),
            "std_rms"    : float(np.nanstd(rms_vals)),
        }

    def compute_jerkiness_per_tap(self, signal, tap_indices, fs, n_hop=1):
        sig_diff  = np.diff(signal)
        jerk_vals = []
        for tap in tap_indices:
            start    = int(tap[0]); end = int(tap[-1])
            if start >= end or end > len(sig_diff):
                continue
            seg      = sig_diff[start:end]
            duration = (end - start) / fs
            if duration <= 0:
                continue
            count = sum(1 for i in range(len(seg) - n_hop)
                        if seg[i + n_hop] * seg[i] < 0)
            jerk_vals.append(count / duration)
        jerk_arr = np.array(jerk_vals)
        return {
            "jerkiness_per_tap": jerk_arr.tolist(),
            "mean_jerkiness"   : float(np.nanmean(jerk_arr)),
            "std_jerkiness"    : float(np.nanstd(jerk_arr)),
        }

    def compute_entropy_per_tap(self, signal, tap_indices):
        entr_vals = []
        for tap in tap_indices:
            start = int(tap[0]); end = int(tap[-1])
            if start >= end or end > len(signal):
                continue
            entr_vals.append(_calc_entropy(signal[start:end]))
        entr_arr = np.array(entr_vals)
        return {
            "entropy_per_tap": entr_arr.tolist(),
            "mean_entropy"   : float(np.nanmean(entr_arr)),
            "std_entropy"    : float(np.nanstd(entr_arr)),
        }

    def compute_velocity_raising_auc(self, signal, tap_indices):
        auc_vals = []
        for tap in tap_indices:
            start = int(tap[0]); mid = int(tap[1])
            if start >= mid or mid > len(signal):
                continue
            line  = signal[start:mid]
            areas = [np.mean([line[s + 1], line[s]]) for s in range(len(line) - 1)]
            auc_vals.append(float(sum(areas)))
        auc_arr = np.array(auc_vals)
        return {
            "raising_auc_per_tap": auc_arr.tolist(),
            "mean_raising_auc"   : float(np.nanmean(auc_arr)),
            "std_raising_auc"    : float(np.nanstd(auc_arr)),
        }

    @staticmethod
    def extract_tap_waveforms(signal, tap_indices, n_points=200):
        waveforms = []
        common_x  = np.linspace(0, 1, n_points)
        for tap in tap_indices:
            start = int(tap[0]); end = int(tap[-1])
            if start >= end or end > len(signal):
                waveforms.append(np.full(n_points, np.nan))
                continue
            seg    = signal[start:end]
            orig_x = np.linspace(0, 1, len(seg))
            waveforms.append(np.interp(common_x, orig_x, seg))
        return waveforms


# -----------------------------------------------------------------------------
def build_summary_table(hand_name, tap_features, envelope_features,
                        cycle_data, hb_features, distance_metrics, accel_features):
    v_open  = np.array(tap_features["opening_velocity"])
    v_close = np.array(tap_features["closing_velocity"])
    rows = [
        ("Mean peak distance (ratio)",            f"{distance_metrics['mean_distance']:.4f}"),
        ("Max peak distance (ratio)",             f"{distance_metrics['max_distance']:.4f}"),
        ("Min peak distance (ratio)",             f"{distance_metrics['min_distance']:.4f}"),
        ("Std peak distance (ratio)",             f"{distance_metrics['std_distance']:.4f}"),
        ("Total taps (M)",                        f"{hb_features['M']}"),
        ("Amp_avg - Average amplitude (ratio)",   f"{hb_features['amp_avg']:.4f}"),
        ("Amp_avg - Average amplitude (% hand)",  f"{hb_features['amp_avg_pct']:.2f}%"),
        ("CD_avg  - Average cycle duration (s)",  f"{hb_features['cd_avg']:.4f}s"),
        ("Std cycle duration (s)",                f"{cycle_data['std_cycle_dur']:.4f}s"),
        ("Min ITI (s)",                           f"{cycle_data['min_iti']:.4f}s"),
        ("Max ITI (s)",                           f"{cycle_data['max_iti']:.4f}s"),
        ("CV_ITI  - Rhythm regularity",           f"{cycle_data['cv_iti']:.4f}"),
        ("Cycle slope (s/cycle)",                 f"{cycle_data['cycle_slope']:.4e}"),
        ("Amplitude slope (%/s)",                 f"{envelope_features['amplitude_slope']:.4f}"),
        ("Early mean amplitude (%)",              f"{envelope_features['early_mean_amplitude']:.2f}%"),
        ("Late mean amplitude (%)",               f"{envelope_features['late_mean_amplitude']:.2f}%"),
        ("Total amplitude decay (%)",             f"{envelope_features['total_amplitude_decay']:.2f}%"),
        ("Decay percent (%)",                     f"{envelope_features['decay_percent']:.2f}%"),
        ("Mean opening velocity (ratio/s)",       f"{float(np.nanmean(v_open)):.4f}"),
        ("Std opening velocity (ratio/s)",        f"{float(np.nanstd(v_open)):.4f}"),
        ("Mean closing velocity (ratio/s)",       f"{float(np.nanmean(v_close)):.4f}"),
        ("Std closing velocity (ratio/s)",        f"{float(np.nanstd(v_close)):.4f}"),
        ("Mean peak acceleration (ratio/s2)",     f"{accel_features['mean_peak_accel']:.4f}"),
        ("Std  peak acceleration (ratio/s2)",     f"{accel_features['std_peak_accel']:.4f}"),
        ("Peak acceleration slope (ratio/s2/s)",  f"{accel_features['peak_accel_slope']:.4e}"),
        ("Mean opening acceleration (ratio/s2)",  f"{accel_features['mean_open_accel']:.4f}"),
        ("Std  opening acceleration (ratio/s2)",  f"{accel_features['std_open_accel']:.4f}"),
        ("Mean closing acceleration (ratio/s2)",  f"{accel_features['mean_close_accel']:.4f}"),
        ("Std  closing acceleration (ratio/s2)",  f"{accel_features['std_close_accel']:.4f}"),
    ]
    df = pd.DataFrame(rows, columns=["Feature", "Value"])
    df.insert(0, "Hand", hand_name)
    return df


# -----------------------------------------------------------------------------
def extract_hand_features(extractor, results, fe, hand_name, fs=30, y_max=None):
    if hand_name not in results or len(results[hand_name]["peaks"]) < 3:
        return None

    signal     = extractor.right_dist_normalized if hand_name == "Right" \
                 else extractor.left_dist_normalized
    time_array = extractor.right_hand_time if hand_name == "Right" \
                 else extractor.left_hand_time

    peaks           = results[hand_name]["peaks"]
    tap_start_frame = results[hand_name]["tap_start_frame"]
    tap_end_frame   = results[hand_name]["tap_end_frame"]

    tap_features      = fe.compute_tap_features(signal, peaks, time_array,
                                                hand_name, tap_start_frame, tap_end_frame)
    envelope_features = fe.compute_envelope_features(tap_features)
    cycle_data        = fe.compute_cycle_duration(tap_features)
    hb_features       = fe.compute_hypokinesia_bradykinesia(tap_features)
    distance_metrics  = fe.compute_distance_metrics(signal, peaks)
    accel_features    = fe.compute_acceleration_features(tap_features)
    velocity, acceleration = fe.compute_velocity_acceleration(signal, time_array)
    speed             = fe.compute_speed(signal, time_array)

    tap_indices = fe.build_tap_indices(
        tap_features["validated_peaks"],
        tap_features["left_valley_idxs"],
        tap_features["right_valley_idxs"],
    )

    rms_features     = fe.compute_rms_per_tap(signal, tap_indices, fs)
    jerk_features    = fe.compute_jerkiness_per_tap(signal, tap_indices, fs)
    entropy_features = fe.compute_entropy_per_tap(signal, tap_indices)
    raising_auc      = fe.compute_velocity_raising_auc(signal, tap_indices)
    tap_waveforms    = fe.extract_tap_waveforms(signal, tap_indices)

    summary_table = build_summary_table(hand_name, tap_features, envelope_features,
                                        cycle_data, hb_features, distance_metrics,
                                        accel_features)
    return {
        "distance_metrics"   : distance_metrics,
        "tap_features"       : tap_features,
        "envelope_features"  : envelope_features,
        "cycle_data"         : cycle_data,
        "hb_features"        : hb_features,
        "accel_features"     : accel_features,
        "rms_features"       : rms_features,
        "jerk_features"      : jerk_features,
        "entropy_features"   : entropy_features,
        "raising_auc"        : raising_auc,
        "tap_waveforms"      : tap_waveforms,
        "tap_indices"        : tap_indices,
        "velocity_signal"    : velocity,
        "acceleration_signal": acceleration,
        "speed_signal"       : speed,
        "summary_table"      : summary_table,
    }


