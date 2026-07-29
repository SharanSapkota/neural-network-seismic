
import numpy as np
from earth_model import FS, FREQ_MIN, FREQ_MAX

def add_noise(clean, noise_level, rng):
    noise = rng.normal(0.0, noise_level, size=len(clean))
    noisy = clean + noise
    return noisy.astype(np.float32), np.float32(noise_level)

def _bandpass(signal, fs, f_lo, f_hi):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum[(freqs < f_lo) | (freqs > f_hi)] = 0.0
    return np.fft.irfft(spectrum, n=len(signal))

def add_realistic_noise(clean, noise_level, rng,
                        n_spurious=(0, 3), fs=FS,
                        f_lo=FREQ_MIN, f_hi=FREQ_MAX):

    n = len(clean)

    white = rng.normal(0.0, 1.0, size=n)
    band = _bandpass(white, fs, f_lo, f_hi)

    spurious = np.zeros(n)
    count = rng.integers(n_spurious[0], n_spurious[1] + 1)
    lag = np.arange(n)
    for _ in range(count):
        centre = rng.integers(0, n)
        freq = rng.uniform(f_lo, f_hi)
        width = rng.uniform(20, 60)
        env = np.exp(-0.5 * ((lag - centre) / width) ** 2)
        spurious += env * np.sin(2 * np.pi * freq * (lag - centre) / fs)

    combined = band + 0.5 * spurious

    combined = combined - combined.mean()
    current_std = combined.std() + 1e-12
    combined = combined * (noise_level / current_std)

    noisy = clean + combined
    return noisy.astype(np.float32), np.float32(noise_level)


def pick_noise_level(rng, level_min=0.05, level_max=0.80):
    """Random noise level inside the training range."""
    return rng.uniform(level_min, level_max)