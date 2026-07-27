
import numpy as np
from disba import PhaseDispersion, GroupDispersion

FS = 500.0
N_SAMPLES = 1001
FREQ_MIN = 3.0
FREQ_MAX = 50.0
N_FREQ = 40

def dam_model():
    thickness_m   = np.array([20.0, 40.0, 60.0, 0.0])
    vs_m_s        = np.array([400.0, 250.0, 150.0, 800.0])
    density_g_cm3 = np.array([2.0, 1.9, 1.8, 2.2])

    thickness_km = thickness_m / 1000.0
    vs_km_s = vs_m_s / 1000.0
    vp_km_s = vs_km_s * 1.73

    return np.column_stack([thickness_km, vp_km_s, vs_km_s, density_g_cm3])


def true_dispersion(model):
    periods = np.linspace(1.0 / FREQ_MAX, 1.0 / FREQ_MIN, N_FREQ)

    pd = PhaseDispersion(*model.T)
    res = pd(periods, mode=0, wave='rayleigh')

    if len(res.period) < N_FREQ:
        print(f'  WARNING: requested {N_FREQ} periods, disba returned '
              f'{len(res.period)} (velocity inversion may be dropping modes)')

    freqs = 1.0 / res.period
    vels = res.velocity * 1000.0
    order = np.argsort(freqs)
    # Returns which frequency travels at what speed.
    return freqs[order], vels[order]

def true_group_velocity(model, freqs):
    periods = 1.0 / freqs
    periods = np.sort(periods)

    gd = GroupDispersion(*model.T)
    res = gd(periods, mode=0, wave='rayleigh')

    group_freqs = 1.0 / res.period
    group_vels = res.velocity * 1000.0
    order = np.argsort(group_freqs)

    return np.interp(freqs, group_freqs[order], group_vels[order])


def synth_ccf(freqs, phase_vels, group_vels, distance_m):
    lag = (np.arange(N_SAMPLES) - N_SAMPLES // 2) / FS 
    lag_abs = np.abs(lag)  

    sig = np.zeros(N_SAMPLES)

    for f, v_phase, v_group in zip(freqs, phase_vels, group_vels):
        t_group = distance_m / v_group # Travel times (How long does the wave energy take to travel from Sensor A to Sensor B?)
        t_phase = distance_m / v_phase

        width = 2.0 / f  # How long does the wave last?
        envelope = np.exp(-0.5 * ((lag_abs - t_group) / width) ** 2) # At which time should the wave energy be strongest? Gaussian envelope
        oscillation = np.sin(2.0 * np.pi * f * (lag_abs - t_phase)) # It simply asnwers What does the wave look like? sinusoidal oscillation

        sig += envelope * oscillation

    sig = sig - sig.mean()
    return (sig / (sig.std() + 1e-12)).astype(np.float32) # normalize