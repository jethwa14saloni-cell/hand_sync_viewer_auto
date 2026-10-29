"""Offline, local finger-tapping activity heuristic (not a clinical classifier)."""
import numpy as np
from scipy.signal import find_peaks, detrend
from scipy.ndimage import gaussian_filter1d

STEP = 0.25
WINDOW = 2.0
MIN_AMPLITUDE = 0.10  # thumb/index distance divided by palm reference
MIN_SPEED = 0.10
MAX_BRIDGE = 0.5


def activity_score(time, signal, center):
    time, signal = np.asarray(time), np.asarray(signal)
    mask = (time >= center-WINDOW/2) & (time <= center+WINDOW/2)
    t, y = time[mask], signal[mask]
    if len(y) < 10 or t[-1]-t[0] < 0.8 or np.mean(np.isfinite(y)) < .75:
        return 0.0
    finite = np.isfinite(y)
    if not np.all(finite):
        runs = np.diff(np.r_[False, ~finite, False].astype(int))
        longest = int(np.max(np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)))
        if longest * float(np.median(np.diff(t))) > MAX_BRIDGE:
            return 0.0
        y = np.interp(t, t[finite], y[finite])
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0
    y = gaussian_filter1d(y, max(.5, .035/dt))
    y = detrend(y)
    amplitude = float(np.percentile(y,90)-np.percentile(y,10))
    speed = float(np.median(np.abs(np.diff(y))/np.diff(t)))
    if amplitude < MIN_AMPLITUDE or speed < MIN_SPEED:
        return 0.0
    # Require repeated opening/closing, not just a single postural change.
    opts = dict(prominence=max(.06, amplitude*.2), distance=max(1,int(.12/dt)))
    maxima = find_peaks(y, **opts)[0]
    minima = find_peaks(-y, **opts)[0]
    if min(len(maxima),len(minima)) < 1 or len(maxima)+len(minima) < 3:
        return 0.0
    return amplitude * speed


class ActivityTimeline:
    def __init__(self, streams):
        self.streams = streams
        ends = [t[-1] for t,y in streams.values() if len(t)]
        self.times = np.arange(0, (max(ends) if ends else 0)+STEP, STEP)
        self.scores = {side: np.array([activity_score(t,y,c) for c in self.times])
                       for side,(t,y) in streams.items()}
        self.labels, self.hands = [], []
        # Initial display uses the hand with greatest aggregate supported activity.
        totals = {k:float(np.sum(v)) for k,v in self.scores.items()}
        current = max(totals,key=totals.get) if totals and max(totals.values())>0 else None
        pending, count = None, 0
        for i in range(len(self.times)):
            values = {k:float(v[i]) for k,v in self.scores.items()}
            active = [k for k,v in values.items() if v>0]
            candidate = max(active,key=values.get) if active else None
            both = len(active)==2 and max(values.values()) < 1.7*min(values.values())
            if candidate and not both:
                count = count+1 if candidate==pending else 1
                pending = candidate
                if current is None or count>=3:  # 0.75 s confirmation
                    current = candidate
            else:
                pending,count=None,0
            self.hands.append(current)
            self.labels.append('Both hands active' if both else
                (f'{candidate} active' if candidate else 'No clear tapping activity'))

    def at(self, seconds):
        i = int(np.clip(np.searchsorted(self.times,seconds,side='right')-1,0,len(self.times)-1))
        return self.hands[i], self.labels[i]

    def supported(self, hand, seconds):
        values = self.scores.get(hand)
        if values is None:
            return False
        i = int(np.clip(np.searchsorted(self.times,seconds),0,len(self.times)-1))
        return bool(values[i]>0)