"""Real ECG noise from the MIT-BIH Noise Stress Test Database (NSTDB).

The bw (baseline wander), em (electrode motion) and ma (muscle artifact)
records were made on physically active volunteers with electrodes placed
where their own ECG isn't visible, so they are pure recorded noise.

    Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia
    detectors. Computers in Cardiology 1984; 11:381-384.
    https://physionet.org/content/nstdb/1.0.0/  (ODC Attribution License 1.0)

The records are downloaded on first use into data/nstdb/.
"""

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

URL = "https://physionet.org/files/nstdb/1.0.0/"
DATA_DIR = Path(__file__).resolve().parent / "data" / "nstdb"
RECORDS = ("bw", "em", "ma")


def _fetch(record):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("hea", "dat"):
        path = DATA_DIR / f"{record}.{ext}"
        if not path.exists():
            print(f"Downloading {URL}{record}.{ext}")
            urllib.request.urlretrieve(f"{URL}{record}.{ext}", path)


def _read_212(path, n_channels):
    """Decode WFDB format 212: two 12-bit two's-complement samples per 3 bytes."""
    b = np.fromfile(path, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
    s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    samples = np.column_stack([s0, s1]).ravel()
    samples[samples > 2047] -= 4096
    return samples.reshape(-1, n_channels)


@lru_cache(maxsize=None)
def load(record):
    """Return (fs, samples[n, channels]) in mV for one NSTDB noise record."""
    _fetch(record)
    lines = (DATA_DIR / f"{record}.hea").read_text().split("\n")
    _, n_ch, fs, n = lines[0].split()[:4]
    n_ch, fs, n = int(n_ch), float(fs), int(n)
    digital = _read_212(DATA_DIR / f"{record}.dat", n_ch)[:n]

    out = np.empty(digital.shape)
    for ch, line in enumerate(lines[1:1 + n_ch]):
        fields = line.split()
        gain = float(fields[2].split("(")[0].split("/")[0]) or 200.0   # 0 means default
        baseline = int(fields[4])                                     # ADC zero
        if digital[0, ch] != int(fields[5]):
            raise ValueError(f"{record}: first sample of channel {ch} doesn't match header")
        out[:, ch] = (digital[:, ch] - baseline) / gain
    return fs, out


def segment(record, rng, duration, fs_out):
    """A random zero-mean segment of one channel, resampled to fs_out."""
    fs, data = load(record)
    n_in = int(np.ceil(duration * fs)) + 1
    ch = rng.integers(data.shape[1])
    start = rng.integers(0, data.shape[0] - n_in)
    x = data[start:start + n_in, ch]
    up, down = int(fs_out), int(fs)
    g = np.gcd(up, down)
    y = signal.resample_poly(x, up // g, down // g)[:int(round(duration * fs_out))]
    return y - y.mean()
