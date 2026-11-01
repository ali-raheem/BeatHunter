"""Pan-Tompkins QRS detector, used as an established baseline.

Follows Pan & Tompkins, "A Real-Time QRS Detection Algorithm", IEEE Trans.
Biomed. Eng. 32(3), 1985: band-pass 5-15 Hz, derivative, squaring, 150 ms
moving-window integration, then adaptive signal/noise thresholds with a 200 ms
refractory period, T-wave discrimination, and searchback for missed beats.

Simplifications: the filters are zero-phase (offline), and only the
integrated-signal thresholds are used. The paper also runs a second, parallel
set of thresholds on the band-passed signal.

Addition: if no beat is found for 3 s, the thresholds are re-learned from the
last 2 s, as practical implementations do. Without this, one large transient
(e.g. a filter start-up edge) can set the signal level so high that nothing
is ever detected again.
"""

import numpy as np
from scipy import signal


def detect(x, fs):
    """Return sample indices of detected beats (peaks of the integrated signal)."""
    b, a = signal.butter(2, [5, 15], btype="band", fs=fs)
    bp = signal.filtfilt(b, a, x)
    deriv = np.convolve(bp, [1, 2, 0, -2, -1], mode="same") * fs / 8
    n_mwi = max(1, int(round(0.15 * fs)))
    mwi = np.convolve(deriv ** 2, np.ones(n_mwi) / n_mwi, mode="same")

    refractory = int(round(0.2 * fs))
    peaks, _ = signal.find_peaks(mwi, distance=refractory)

    def learn(end):
        seg = mwi[max(0, end - int(2 * fs)):end]
        return 0.25 * seg.max(), 0.5 * seg.mean()

    # Skip the first 0.5 s, where filter edge effects live.
    spki, npki = learn(int(2.5 * fs))
    last_event = 0
    restarted = False           # don't measure an RR interval across a re-learn
    qrs, rr = [], []
    rr_avg2 = None
    last_slope = 0.0
    noise_peaks = []

    def slope(p):
        return np.abs(deriv[max(0, p - n_mwi):p + 1]).max()

    def accept(p, weight):
        nonlocal spki, rr_avg2, last_slope, restarted
        spki = weight * mwi[p] + (1 - weight) * spki
        if qrs and not restarted:
            interval = p - qrs[-1]
            rr.append(interval)
            recent = rr[-8:]
            avg1 = np.mean(recent)
            if rr_avg2 is None:
                rr_avg2 = avg1
            regular = [r for r in recent if 0.92 * rr_avg2 <= r <= 1.16 * rr_avg2]
            rr_avg2 = np.mean(regular) if regular else avg1
        qrs.append(p)
        restarted = False
        last_slope = slope(p)

    for p in peaks:
        thr1 = npki + 0.25 * (spki - npki)
        thr2 = 0.5 * thr1
        if rr and not (0.92 * rr_avg2 <= rr[-1] <= 1.16 * rr_avg2):
            thr1, thr2 = 0.5 * thr1, 0.5 * thr2       # irregular rhythm

        # Searchback: no beat for 166% of the average RR interval (1 s before
        # an average exists).
        since = qrs[-1] if qrs else 0
        expected = rr_avg2 if rr_avg2 is not None else fs
        if p - since > 1.66 * expected:
            missed = [q for q in noise_peaks if q - since > refractory and mwi[q] > thr2]
            if missed:
                accept(max(missed, key=lambda q: mwi[q]), 0.25)
                last_event = qrs[-1]
            noise_peaks = [q for q in noise_peaks if qrs and q > qrs[-1]]

        # Lost: re-learn the thresholds from the recent signal.
        if p - max(last_event, qrs[-1] if qrs else 0) > 3 * fs:
            spki, npki = learn(p)
            rr_avg2, rr, last_event, restarted = None, [], p, True
            thr1 = npki + 0.25 * (spki - npki)

        if mwi[p] > thr1:
            # T-wave check: an early peak with a shallow slope is a T wave.
            if qrs and p - qrs[-1] < 0.36 * fs and slope(p) < 0.5 * last_slope:
                npki = 0.125 * mwi[p] + 0.875 * npki
                noise_peaks.append(p)
                continue
            accept(p, 0.125)
            noise_peaks = []
        else:
            npki = 0.125 * mwi[p] + 0.875 * npki
            noise_peaks.append(p)

    return np.array(qrs, dtype=int)


def heart_rate(beat_times, frame_times, window_sec):
    """BPM at each frame from the median RR interval of beats inside a window
    centred on the frame (the same window the FFT tracker sees). Frames with
    fewer than two beats hold the previous value."""
    out = np.full(len(frame_times), np.nan)
    last = np.nan
    for i, tc in enumerate(frame_times):
        inside = beat_times[np.abs(beat_times - tc) <= window_sec / 2]
        if inside.size >= 2:
            last = 60 / np.median(np.diff(inside))
        out[i] = last
    return out
