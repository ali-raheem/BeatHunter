"""Adaptive FFT heartbeat tracker prototype.

Tracks a time-varying heart rate through a noisy synthetic signal using:
  * a sliding-window FFT with exponential power-spectrum averaging,
  * a Gaussian frequency mask centred on the tracked peak,
  * a three-state confidence machine (SEARCHING / LOCKED / REACQUIRING) that
    adapts both the mask's centre frequency and its bandwidth,
  * inverse-FFT overlap-add reconstruction of the filtered heartbeat.

The result is compared with a naive "strongest peak in band" estimator against
the known synthetic heart rate.

Usage:
    python adaptive_heartbeat_tracker.py [--seed N] [--out results.png] [--show] [--no-plot]
"""

import argparse
from dataclasses import dataclass, field

import numpy as np

# Signal / analysis parameters
FS = 50.0                # sample rate (Hz)
DURATION = 120.0         # seconds
WIN_SEC = 8.0            # FFT window length
HOP_SEC = 0.5            # hop between frames
NFFT = 8192              # zero-padded FFT size for peak resolution
BAND = (0.7, 3.5)        # plausible heart-rate band (Hz) = 42-210 BPM
PSD_ALPHA = 0.25         # exponential averaging weight for new spectra
WARMUP_SEC = 10.0        # excluded from error metrics

# Tracker parameters
SIGMA_LOCKED = 0.12      # mask std-dev when locked (Hz)
SIGMA_MAX = 1.0          # mask std-dev when searching (Hz)
SIGMA_GROWTH = 1.25      # per-frame mask widening while reacquiring
MAX_STEP_HZ = 0.05       # max centre-frequency change per hop when locked
CONF_LOCK = 6.0          # confidence needed to lock
CONF_LOSE = 3.0          # confidence below which a lock is in doubt
LOCK_FRAMES = 4          # consistent frames needed to lock
LOSE_FRAMES = 3          # weak frames before reacquiring
PERSIST_TOL_HZ = 0.08    # candidate stability tolerance while searching

SEARCHING, LOCKED, REACQUIRING = "SEARCHING", "LOCKED", "REACQUIRING"
STATE_CODE = {SEARCHING: 0, REACQUIRING: 1, LOCKED: 2}


def true_hr_bpm(t):
    """Ground-truth heart rate: a slow rise and fall plus a small wobble."""
    return 70 + 45 * np.sin(np.pi * t / DURATION) ** 2 + 3 * np.sin(2 * np.pi * t / 17)


def synthesize(seed):
    """Return (t, noisy, clean_heartbeat) for a 120 s synthetic recording."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, 1 / FS)
    f_hr = true_hr_bpm(t) / 60
    phase = 2 * np.pi * np.cumsum(f_hr) / FS

    # Pulse-like waveform: fundamental plus a weaker second harmonic.
    clean = np.sin(phase) + 0.45 * np.sin(2 * phase + 0.6)

    noise = 1.6 * rng.standard_normal(t.size)

    # Competing rhythmic source (e.g. periodic motion) that is stronger than
    # the heartbeat while active, with soft on/off ramps.
    env = np.clip((t - 40) / 3, 0, 1) * np.clip((78 - t) / 3, 0, 1)
    competitor = 2.0 * env * np.sin(2 * np.pi * 1.3 * t)

    # Short broadband motion artifacts.
    artifacts = np.zeros_like(t)
    for start in (25.0, 92.0):
        idx = (t >= start) & (t < start + 1.5)
        artifacts[idx] = 6 * rng.standard_normal(idx.sum())

    # Sensor contact loss: the heartbeat disappears for a few seconds, which
    # should drive the tracker out of LOCKED and through REACQUIRING.
    contact = 1 - np.clip((t - 95) / 1, 0, 1) * np.clip((104 - t) / 1, 0, 1)

    return t, contact * clean + noise + competitor + artifacts, clean


def gaussian(f, centre, sigma):
    return np.exp(-0.5 * ((f - centre) / sigma) ** 2)


@dataclass
class Tracker:
    freqs: np.ndarray
    band: np.ndarray
    state: str = SEARCHING
    centre: float = 1.5
    sigma: float = SIGMA_MAX
    candidate: float = np.nan
    streak: int = 0
    weak: int = 0
    history: list = field(default_factory=list)

    def confidence(self, psd, f):
        """Peak power at f relative to the in-band median (a peak SNR)."""
        k = np.argmin(np.abs(self.freqs - f))
        return psd[k] / (np.median(psd[self.band]) + 1e-12)

    def subharmonic_check(self, psd, f):
        """Prefer f/2 if it carries comparable power (we picked a harmonic)."""
        half = f / 2
        if half >= BAND[0] and self.confidence(psd, half) > 0.5 * self.confidence(psd, f):
            return half
        return f

    def masked_peak(self, psd):
        weighted = psd * gaussian(self.freqs, self.centre, self.sigma)
        weighted[~self.band] = 0
        return self.freqs[np.argmax(weighted)]

    def step(self, psd):
        if self.state == SEARCHING:
            f = psd.copy()
            f[~self.band] = 0
            peak = self.subharmonic_check(psd, self.freqs[np.argmax(f)])
            conf = self.confidence(psd, peak)
            if abs(peak - self.candidate) < PERSIST_TOL_HZ and conf > CONF_LOCK:
                self.streak += 1
            else:
                self.streak = 0
            self.candidate = peak
            self.centre = peak
            if self.streak >= LOCK_FRAMES:
                self.state, self.sigma, self.weak = LOCKED, SIGMA_LOCKED, 0

        else:
            peak = self.masked_peak(psd)
            conf = self.confidence(psd, peak)
            if self.state == LOCKED:
                # Temporal prior: limit how far the estimate may move per hop.
                self.centre += np.clip(peak - self.centre, -MAX_STEP_HZ, MAX_STEP_HZ)
                self.weak = self.weak + 1 if conf < CONF_LOSE else 0
                if self.weak >= LOSE_FRAMES:
                    self.state = REACQUIRING
            else:  # REACQUIRING: widen gradually, re-lock if confidence returns
                self.centre = peak
                if conf > CONF_LOCK:
                    self.state, self.sigma, self.weak = LOCKED, SIGMA_LOCKED, 0
                else:
                    self.sigma = min(self.sigma * SIGMA_GROWTH, SIGMA_MAX)
                    if self.sigma >= SIGMA_MAX:
                        self.state, self.streak, self.candidate = SEARCHING, 0, np.nan

        self.history.append((self.centre, self.sigma, self.state, conf))
        return self.centre


def run(seed):
    return track(*synthesize(seed))


def track(t, x, clean, hr_fn=true_hr_bpm):
    """Track a signal sampled at FS; hr_fn(t) gives the true rate in BPM."""
    win = int(WIN_SEC * FS)
    hop = int(HOP_SEC * FS)
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    seg_freqs = np.fft.rfftfreq(win, 1 / FS)

    tracker = Tracker(freqs=freqs, band=band)
    avg_psd = None
    frame_t, est, naive, psds = [], [], [], []
    recon = np.zeros_like(x)
    norm = np.zeros_like(x)

    for start in range(0, x.size - win + 1, hop):
        seg = x[start:start + win]
        seg = (seg - seg.mean()) * window
        psd = np.abs(np.fft.rfft(seg, NFFT)) ** 2
        avg_psd = psd if avg_psd is None else PSD_ALPHA * psd + (1 - PSD_ALPHA) * avg_psd

        f0 = tracker.step(avg_psd)
        in_band = np.where(band, avg_psd, 0)

        frame_t.append(t[start + win // 2])
        est.append(f0)
        naive.append(freqs[np.argmax(in_band)])
        psds.append(avg_psd[band])

        # Reconstruct: mask the frame's own spectrum around f0 and 2*f0.
        sigma = tracker.sigma
        mask = np.clip(gaussian(seg_freqs, f0, sigma) + gaussian(seg_freqs, 2 * f0, sigma), 0, 1)
        filtered = np.fft.irfft(np.fft.rfft(seg) * mask, win)
        recon[start:start + win] += filtered * window
        norm[start:start + win] += window ** 2

    # Ignore the ramp-up/down at the edges where few windows overlap.
    covered = norm > 0.1 * norm.max()
    recon = np.divide(recon, norm, out=np.zeros_like(recon), where=covered)

    frame_t = np.array(frame_t)
    truth = hr_fn(frame_t)
    est_bpm = np.array(est) * 60
    naive_bpm = np.array(naive) * 60
    valid = frame_t >= WARMUP_SEC

    corr = np.corrcoef(recon[covered], clean[covered])[0, 1]

    return dict(
        t=t, x=x, clean=clean, recon=recon, frame_t=frame_t, truth=truth,
        est_bpm=est_bpm, naive_bpm=naive_bpm, valid=valid,
        mae_tracker=np.mean(np.abs(est_bpm - truth)[valid]),
        mae_naive=np.mean(np.abs(naive_bpm - truth)[valid]),
        recon_corr=corr,
        psds=np.array(psds), band_bpm=freqs[band] * 60,
        history=tracker.history,
    )


def plot(r, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1, 2]})

    ax = axes[0]
    db = 10 * np.log10(r["psds"].T + 1e-12)
    ax.pcolormesh(r["frame_t"], r["band_bpm"], db, shading="auto", cmap="magma",
                  vmin=np.percentile(db, 30), vmax=db.max())
    ax.plot(r["frame_t"], r["truth"], "c-", lw=2, label="True HR")
    ax.plot(r["frame_t"], r["est_bpm"], "w--", lw=1.8,
            label=f"Adaptive tracker (MAE {r['mae_tracker']:.2f} BPM)")
    ax.plot(r["frame_t"], r["naive_bpm"], "g.", ms=3,
            label=f"Naive strongest peak (MAE {r['mae_naive']:.2f} BPM)")
    ax.axvspan(0, WARMUP_SEC, color="grey", alpha=0.3)
    ax.set_ylabel("BPM")
    ax.set_title("Averaged power spectrum with tracked heart rate")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    codes = [STATE_CODE[h[2]] for h in r["history"]]
    ax.step(r["frame_t"], codes, where="mid", color="tab:blue")
    ax.set_yticks(list(STATE_CODE.values()), list(STATE_CODE.keys()))
    ax2 = ax.twinx()
    ax2.plot(r["frame_t"], [h[1] * 60 for h in r["history"]], color="tab:orange", alpha=0.7)
    ax2.set_ylabel("Mask σ (BPM)", color="tab:orange")
    ax.set_title("Tracker state and mask bandwidth")

    ax = axes[2]
    zoom = (r["t"] >= 50) & (r["t"] < 60)
    ax.plot(r["t"][zoom], r["x"][zoom], color="lightgrey", lw=0.8, label="Noisy input")
    ax.plot(r["t"][zoom], r["clean"][zoom], "k", lw=1.5, label="Clean heartbeat")
    ax.plot(r["t"][zoom], r["recon"][zoom], "r--", lw=1.5,
            label=f"Reconstruction (r = {r['recon_corr']:.2f} overall)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Reconstruction, 50-60 s (competing source active)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved plot to {out}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="adaptive_heartbeat_tracker.png")
    p.add_argument("--show", action="store_true", help="open an interactive plot window")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    r = run(args.seed)
    print(f"Adaptive tracker MAE : {r['mae_tracker']:.2f} BPM")
    print(f"Naive peak MAE       : {r['mae_naive']:.2f} BPM")
    print(f"Reconstruction corr. : {r['recon_corr']:.3f}")
    if not args.no_plot:
        plot(r, args.out, args.show)


if __name__ == "__main__":
    main()
