"""Run the adaptive tracker against a set of noise scenarios.

Signals are synthesised at a realistic ADC rate (500 Hz) with a QRS-shaped
ECG, then anti-alias filtered and decimated to the tracker's 50 Hz, which is
roughly what a real front end would do. Each scenario runs over several seeds
and the table reports the mean across seeds.

Three methods are compared, all on the same 50 Hz signal:
  raw  - the FFT tracker runs directly on the decimated ECG.
  qrs  - the ECG is first turned into a QRS energy envelope (band-pass
         5-20 Hz, square, 0.15 s moving average), so each beat becomes one
         broad bump whose fundamental is the heart rate.
  pt   - Pan-Tompkins beat detection, with heart rate taken from the median RR
         interval inside the same 8 s window the tracker uses.

Scenarios come in two groups: synthetic noise models, and real recorded noise
from the MIT-BIH Noise Stress Test Database (see nstdb.py).

Usage:
    python noise_scenarios.py [--seeds 5] [--only mains] [--out-dir images] [--no-plot]
"""

import argparse

import numpy as np
from scipy import signal

import adaptive_heartbeat_tracker as aht
import nstdb
import pan_tompkins

FS_RAW = 500.0
DECIMATE = int(FS_RAW / aht.FS)
BACKGROUND = 0.3         # white noise present in every scenario (R peak = 1)


# ---------------------------------------------------------------- heart rate

def hr_workout(t):
    return aht.true_hr_bpm(t)


def hr_sprint(t):
    """Rest, a hard 25 s ramp to 165 BPM, hold, then recover."""
    up = np.clip((t - 30) / 25, 0, 1)
    down = np.clip((t - 80) / 30, 0, 1)
    s = lambda u: u * u * (3 - 2 * u)   # smoothstep
    return 75 + 90 * s(up) - 70 * s(down)


# ---------------------------------------------------------------- components

def ecg(t, hr_fn, rng):
    """QRS-shaped ECG: P, Q, R, S, T Gaussians placed at each beat."""
    phase = np.cumsum(hr_fn(t) / 60) / FS_RAW
    beats = np.flatnonzero(np.diff(np.floor(phase)) > 0)
    impulses = np.zeros_like(t)
    impulses[beats] = 1 + 0.05 * rng.standard_normal(beats.size)
    k = np.arange(-0.35, 0.45, 1 / FS_RAW)
    waves = [(-0.20, 0.025, 0.15), (-0.03, 0.010, -0.10), (0.0, 0.012, 1.0),
             (0.03, 0.010, -0.25), (0.25, 0.050, 0.30)]
    kernel = sum(a * np.exp(-0.5 * ((k - c) / w) ** 2) for c, w, a in waves)
    return np.convolve(impulses, kernel, mode="same")


def white(t, rng, sigma):
    return sigma * rng.standard_normal(t.size)


def mains(t, rng, f, amp):
    """Mains pickup with slow frequency drift, plus a 2nd harmonic (flicker)."""
    drift = 1 + 0.002 * np.sin(2 * np.pi * t / 37 + rng.uniform(0, 6))
    ph = 2 * np.pi * np.cumsum(f * drift) / FS_RAW
    return amp * (np.sin(ph) + 0.3 * np.sin(2 * ph + 1.0))


def wander(t, rng, amp):
    """Breathing (~0.25 Hz) plus slow random baseline drift."""
    b, a = signal.butter(2, 0.3, fs=FS_RAW)
    drift = signal.filtfilt(b, a, rng.standard_normal(t.size))
    drift *= amp / drift.std()
    return amp * 0.7 * np.sin(2 * np.pi * 0.25 * t) + drift


def motion(t, rng, rate, amp):
    """Electrode shifts: random offset steps that decay through the front-end
    high-pass (tau ~1 s), plus short spikes."""
    events = rng.random(t.size) < rate / FS_RAW
    steps = np.where(events, amp * rng.standard_normal(t.size), 0)
    decay = np.exp(-1 / FS_RAW)
    out = signal.lfilter([1], [1, -decay], steps)
    spikes = rng.random(t.size) < rate / FS_RAW
    out += np.where(spikes, 2 * amp * rng.standard_normal(t.size), 0)
    return out


def emg(t, rng, amp, grip_hz=1.3):
    """Muscle noise, 20-150 Hz, modulated by grip effort."""
    b, a = signal.butter(4, [20, 150], btype="band", fs=FS_RAW)
    n = signal.filtfilt(b, a, rng.standard_normal(t.size))
    n /= n.std()
    env = 0.3 + 0.7 * np.sin(np.pi * grip_hz * t) ** 2
    return amp * env * n


def cadence(t, rng, freq_fn, amp):
    """Footstrike / pedal-stroke artifact: a jittered sharp bump each cycle."""
    phase = np.cumsum(freq_fn(t)) / FS_RAW
    hits = np.flatnonzero(np.diff(np.floor(phase)) > 0)
    impulses = np.zeros_like(t)
    impulses[hits] = amp * (1 + 0.2 * rng.standard_normal(hits.size))
    k = np.arange(-0.2, 0.3, 1 / FS_RAW)
    kernel = np.exp(-0.5 * (k / 0.04) ** 2) - 0.4 * np.exp(-0.5 * ((k - 0.1) / 0.06) ** 2)
    return np.convolve(impulses, kernel, mode="same")


def contact_gate(t, rng, n_drop):
    """1 where hands are on the grips, 0 during n_drop random 2-6 s releases."""
    gate = np.ones_like(t)
    for start in np.sort(rng.uniform(15, DURATION_RAW - 10, n_drop)):
        gate[(t >= start) & (t < start + rng.uniform(2, 6))] = 0
    return gate


DURATION_RAW = aht.DURATION


# ---------------------------------------------------------------- scenarios

def scenario(name, hr_fn=hr_workout, group="synthetic"):
    def wrap(fn):
        SCENARIOS.append((name, hr_fn, fn, group))
        return fn
    return wrap


SCENARIOS = []


@scenario("clean ECG")
def _(t, rng, heart):
    return heart


for sigma in (1.0, 2.0, 3.0, 4.0):
    scenario(f"white noise sigma={sigma:g}")(
        lambda t, rng, heart, s=sigma: heart + white(t, rng, s))


@scenario("mains 50+100 Hz x10")
def _(t, rng, heart):
    return heart + mains(t, rng, 50, 10)


@scenario("mains 60+120 Hz x10")
def _(t, rng, heart):
    return heart + mains(t, rng, 60, 10)


@scenario("mains x20 + amp clipping")
def _(t, rng, heart):
    return np.clip(heart + mains(t, rng, 50, 20), -3, 3)


@scenario("baseline wander x3")
def _(t, rng, heart):
    return heart + wander(t, rng, 3)


@scenario("motion steps/spikes")
def _(t, rng, heart):
    return heart + motion(t, rng, rate=0.5, amp=4)


@scenario("EMG (grip) x4")
def _(t, rng, heart):
    return heart + emg(t, rng, 4)


@scenario("cycling cadence 80 rpm")
def _(t, rng, heart):
    return heart + cadence(t, rng, lambda t: np.full_like(t, 80 / 60), 1.5)


@scenario("cadence crossing HR")
def _(t, rng, heart):
    return heart + cadence(t, rng, lambda t: 1.0 + 1.2 * t / aht.DURATION, 1.5)


@scenario("running cadence 165 spm")
def _(t, rng, heart):
    return heart + cadence(t, rng, lambda t: np.full_like(t, 165 / 60), 1.5)


@scenario("intermittent contact x4")
def _(t, rng, heart):
    gate = contact_gate(t, rng, 4)
    # A floating input picks up extra noise while the hands are off.
    return gate * heart + (1 - gate) * white(t, rng, 2.0)


@scenario("sprint HR + white sigma=1", hr_fn=hr_sprint)
def _(t, rng, heart):
    return heart + white(t, rng, 1.0)


@scenario("gym: everything moderate")
def _(t, rng, heart):
    gate = contact_gate(t, rng, 2)
    return (gate * heart + (1 - gate) * white(t, rng, 2.0)
            + white(t, rng, 1.0) + mains(t, rng, 50, 5) + wander(t, rng, 2)
            + motion(t, rng, 0.3, 3) + emg(t, rng, 2)
            + cadence(t, rng, lambda t: 1.4 + 0.3 * np.sin(2 * np.pi * t / 90), 1.0))


def nst_noise(t, rng, heart, records, snr_db):
    """Recorded NSTDB noise scaled to an SNR using the nst convention: signal
    power S = (QRS peak-to-peak amplitude)^2 / 8, noise power N = mean square."""
    noise = sum(nstdb.segment(r, rng, aht.DURATION, FS_RAW) for r in records)
    noise = noise[:t.size]
    s = np.ptp(heart) ** 2 / 8
    return noise * np.sqrt(s / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))


NSTDB_NAMES = {"em": "electrode motion", "ma": "muscle", "bw": "baseline wander"}

for rec, levels in (("em", (12, 6, 0, -6)), ("ma", (12, 6, 0)), ("bw", (0, -6))):
    for db in levels:
        scenario(f"NSTDB {NSTDB_NAMES[rec]} {db:+d} dB", group="nstdb")(
            lambda t, rng, heart, r=rec, d=db: heart + nst_noise(t, rng, heart, [r], d))


@scenario("NSTDB em+ma+bw +6 dB", group="nstdb")
def _(t, rng, heart):
    return heart + nst_noise(t, rng, heart, ["em", "ma", "bw"], 6)


@scenario("NSTDB em+ma+bw +6 dB, sprint", hr_fn=hr_sprint, group="nstdb")
def _(t, rng, heart):
    return heart + nst_noise(t, rng, heart, ["em", "ma", "bw"], 6)


# ---------------------------------------------------------------- evaluation

def front_end(x):
    """Anti-alias low-pass and decimate from FS_RAW to the tracker's FS."""
    return signal.decimate(x, DECIMATE, ftype="fir", zero_phase=True)


def qrs_envelope(x, band=(5, 20), smooth_sec=0.15):
    """QRS energy: most ECG energy is in the 5-20 Hz QRS complex, while the
    0.67-3 Hz band is dominated by harmonics, wander and motion."""
    b, a = signal.butter(3, band, btype="band", fs=aht.FS)
    energy = signal.filtfilt(b, a, x) ** 2
    n = int(smooth_sec * aht.FS)
    return np.convolve(energy, np.ones(n) / n, mode="same")


def evaluate(name, hr_fn, build, seed):
    """Run all three methods on one realisation; returns {method: result}."""
    rng = np.random.default_rng(seed)
    t_raw = np.arange(0, aht.DURATION, 1 / FS_RAW)
    heart = ecg(t_raw, hr_fn, rng)
    x_raw = build(t_raw, rng, heart) + white(t_raw, rng, BACKGROUND)
    t = t_raw[::DECIMATE]
    x, clean = front_end(x_raw), front_end(heart)

    out = {}
    for method, sig in (("raw", x), ("qrs", qrs_envelope(x))):
        r = aht.track(t, sig, clean, hr_fn)
        err = np.abs(r["est_bpm"] - r["truth"])[r["valid"]]
        states = np.array([h[2] for h in r["history"]])
        locked = np.flatnonzero(states == aht.LOCKED)
        r.update(
            name=name,
            within5=np.mean(err <= 5) * 100,
            locked_pct=np.mean(states[r["valid"]] == aht.LOCKED) * 100,
            first_lock=r["frame_t"][locked[0]] if locked.size else np.nan,
        )
        out[method] = r

    frame_t, truth, valid = out["raw"]["frame_t"], out["raw"]["truth"], out["raw"]["valid"]
    beats = pan_tompkins.detect(x, aht.FS)
    est = pan_tompkins.heart_rate(t[beats], frame_t, aht.WIN_SEC)
    err = np.abs(est - truth)[valid]
    err = np.where(np.isnan(err), truth[valid], err)    # no reading yet counts as 0 BPM
    out["pt"] = dict(name=name, frame_t=frame_t, truth=truth, valid=valid, est_bpm=est,
                     mae_tracker=err.mean(), within5=np.mean(err <= 5) * 100)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--only", default="", help="run scenarios whose name contains this text")
    p.add_argument("--out-dir", default="images", help="where the charts are written")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    chosen = [s for s in SCENARIOS if args.only.lower() in s[0].lower()]
    print(f"{'':<32} {'--- raw ---':^15}   {'------- qrs envelope -------':^32}   "
          f"{'- Pan-Tompkins -':^16}")
    header = (f"{'scenario':<32} {'naive':>6} {'tracker':>8}   {'naive':>6} {'tracker':>8} "
              f"{'worst':>6} {'<=5 BPM':>8}   {'MAE':>6} {'<=5 BPM':>8}")
    print(header)
    print("-" * len(header))

    examples = {}
    group_seen = None
    for name, hr_fn, build, group in chosen:
        if group != group_seen and group_seen is not None:
            print()
        group_seen = group
        runs = [evaluate(name, hr_fn, build, s) for s in range(args.seeds)]
        m = lambda meth, k: np.nanmean([r[meth][k] for r in runs])
        worst = max(r["qrs"]["mae_tracker"] for r in runs)
        print(f"{name:<32} {m('raw', 'mae_naive'):>6.1f} {m('raw', 'mae_tracker'):>8.1f}   "
              f"{m('qrs', 'mae_naive'):>6.1f} {m('qrs', 'mae_tracker'):>8.1f} {worst:>6.1f} "
              f"{m('qrs', 'within5'):>7.0f}%   {m('pt', 'mae_tracker'):>6.1f} "
              f"{m('pt', 'within5'):>7.0f}%")
        examples.setdefault(group, []).append(runs[0])

    print("\nMAE in BPM, mean over seeds (worst = worst seed for the QRS tracker). "
          f"'<=5 BPM' is % of frames after the {aht.WARMUP_SEC:g} s warm-up.")

    if not args.no_plot:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        if "synthetic" in examples:
            for method in ("raw", "qrs"):
                plot_grid([r[method] for r in examples["synthetic"]], method,
                          os.path.join(args.out_dir, f"scenarios_{method}.png"))
            plot_compare(examples["synthetic"], "Synthetic noise: QRS tracker vs Pan-Tompkins",
                         os.path.join(args.out_dir, "compare_synthetic.png"))
        if "nstdb" in examples:
            plot_compare(examples["nstdb"], "Recorded NSTDB noise: QRS tracker vs Pan-Tompkins",
                         os.path.join(args.out_dir, "compare_nstdb.png"))
        plot_why(os.path.join(args.out_dir, "why_qrs_envelope.png"))


PIPELINE_TITLES = {
    "raw": "Tracker on the raw ECG",
    "qrs": "Tracker on the QRS energy envelope",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_grid(runs, pipeline, out):
    """One panel per scenario (seed 0): truth, tracker and naive estimates."""
    plt = _plt()
    cols = 3
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.6 * rows + 0.6), sharex=True,
                             squeeze=False)
    for ax, r in zip(axes.flat, runs):
        ax.plot(r["frame_t"], r["naive_bpm"], ".", color="tab:green", ms=2,
                label="naive strongest peak")
        ax.plot(r["frame_t"], r["truth"], color="black", lw=1.5, label="true HR")
        ax.plot(r["frame_t"], r["est_bpm"], color="tab:red", lw=1.2, label="adaptive tracker")
        ax.set_title(f"{r['name']}  (MAE {r['mae_tracker']:.1f}, naive {r['mae_naive']:.1f})",
                     fontsize=9)
        ax.set_ylim(40, 210)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(PIPELINE_TITLES[pipeline], fontsize=13)
    fig.supxlabel("Time (s)")
    fig.supylabel("BPM")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


def plot_compare(runs, title, out):
    """One panel per scenario (seed 0): QRS-envelope tracker against Pan-Tompkins."""
    plt = _plt()
    cols = 3
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.6 * rows + 0.6), sharex=True,
                             squeeze=False)
    for ax, r in zip(axes.flat, runs):
        q, p = r["qrs"], r["pt"]
        ax.plot(p["frame_t"], p["est_bpm"], color="tab:blue", lw=1.1, label="Pan-Tompkins")
        ax.plot(q["frame_t"], q["truth"], color="black", lw=1.5, label="true HR")
        ax.plot(q["frame_t"], q["est_bpm"], color="tab:red", lw=1.2, label="QRS tracker")
        ax.set_title(f"{q['name']}  (tracker {q['mae_tracker']:.1f}, PT {p['mae_tracker']:.1f})",
                     fontsize=9)
        ax.set_ylim(40, 210)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=13)
    fig.supxlabel("Time (s)")
    fig.supylabel("BPM")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


def plot_why(out, bpm=90):
    """Why the raw ECG spectrum misleads and the QRS envelope doesn't."""
    plt = _plt()
    rng = np.random.default_rng(0)
    t_raw = np.arange(0, aht.DURATION, 1 / FS_RAW)
    hr = lambda t: np.full_like(t, bpm, dtype=float)
    heart = ecg(t_raw, hr, rng)
    noisy = heart + wander(t_raw, rng, 3)
    t = t_raw[::DECIMATE]

    def spectrum(x):
        f, p = signal.welch(x, fs=aht.FS, nperseg=int(aht.WIN_SEC * aht.FS), nfft=aht.NFFT)
        band = (f >= aht.BAND[0]) & (f <= aht.BAND[1])
        return f[band] * 60, p[band]

    clean_x, noisy_x = front_end(heart), front_end(noisy)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    ax = axes[0]
    view = (t >= 10) & (t < 14)
    env = qrs_envelope(clean_x)
    ax.plot(t[view], clean_x[view], color="black", lw=1.2, label="ECG")
    ax.plot(t[view], env[view] / env.max(), color="tab:red", lw=1.5,
            label="QRS envelope (scaled)")
    ax.set_title(f"Clean ECG at {bpm} BPM and its QRS envelope", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)

    for ax, x, title in ((axes[1], clean_x, "clean ECG"),
                         (axes[2], noisy_x, "ECG + baseline wander")):
        for sig, color, label in ((x, "black", "raw ECG"),
                                  (qrs_envelope(x), "tab:red", "QRS envelope")):
            f, pw = spectrum(sig)
            ax.plot(f, pw / pw.max(), color=color, lw=1.3, label=label)
        for k in range(1, int(aht.BAND[1] * 60 / bpm) + 1):
            ax.axvline(k * bpm, color="grey", ls=":", lw=1)
            ax.text(k * bpm, 1.02, "HR" if k == 1 else f"{k}×HR", ha="center", fontsize=8)
        ax.set_title(f"Spectrum, {title}", fontsize=10, pad=14)
        ax.set_xlabel("BPM")
        ax.set_ylabel("Relative power")
        ax.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
