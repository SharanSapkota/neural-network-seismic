# data/make_training_pairs.py
import numpy as np
from synthetic_earth_model import (
    dam_model, true_dispersion, true_group_velocity, synth_ccf
)
from data.noise_injector import add_noise


def make_one_pair(distance_m, freqs, phase_vels, group_vels, rng,
                  level_min=0.05, level_max=0.80):
    """clean target + noisy input, for one CCF."""
    clean = synth_ccf(freqs, phase_vels, group_vels, distance_m)
    level = rng.uniform(level_min, level_max)
    noisy, true_level = add_noise(clean, level, rng)

    return noisy, clean, true_level