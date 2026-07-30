
import json
import math
import itertools
import numpy as np

from earth_model import (
    dam_model, true_dispersion, true_group_velocity, synth_ccf
)
from data.noise_injector import add_noise, pick_noise_level


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 


def load_distances(receivers_path):
    receivers = json.load(open(receivers_path))
    ids = sorted(receivers.keys())
    distances = []
    for a, b in itertools.combinations(ids, 2):
        d = haversine(receivers[a]['latitude'], receivers[a]['longitude'],
                      receivers[b]['latitude'], receivers[b]['longitude'])
        if d >= 1:
            distances.append(d)
    return distances

def build_pairs(distances, rng,
                n_noise_per_ccf=5,
                level_min=0.05, level_max=0.80,
                noise_fn=add_noise,
                fixed_level=None):

    model = dam_model()
    freqs, phase_vels = true_dispersion(model)
    group_vels = true_group_velocity(model, freqs)

    noisy_all, clean_all, level_all = [], [], []

    for distance in distances:
        clean = synth_ccf(freqs, phase_vels, group_vels, distance)

        for _ in range(n_noise_per_ccf):
            if fixed_level is not None:
                level = fixed_level
            else:
                level = pick_noise_level(rng, level_min, level_max)

            noisy, true_level = noise_fn(clean, level, rng)

            noisy_all.append(noisy)
            clean_all.append(clean)
            level_all.append(true_level)

    return (np.array(noisy_all, dtype=np.float32),
            np.array(clean_all, dtype=np.float32),
            np.array(level_all, dtype=np.float32)) # (distance X n_noise_per_ccf, 1001) [if distance is 10, 10 X 5, 1001]

def build_out_of_range_test(distances, rng, levels=(0.02, 0.90),
                            noise_fn=add_noise):
    noisy_parts, clean_parts, level_parts = [], [], []
    for level in levels:
        n, c, l = build_pairs(distances, rng,
                              n_noise_per_ccf=1,
                              noise_fn=noise_fn,
                              fixed_level=level)
        noisy_parts.append(n)
        clean_parts.append(c)
        level_parts.append(l)
    return (np.concatenate(noisy_parts),
            np.concatenate(clean_parts),
            np.concatenate(level_parts))