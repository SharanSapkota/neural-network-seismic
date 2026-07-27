
import json
import math
import itertools
import numpy as np

from pair_zarr_store import PairZarrStore
from synthetic_earth_model import (
    dam_model, true_dispersion, true_group_velocity, synth_ccf,
    N_SAMPLES, FS,
)

RECEIVERS = 'data/receivers.json'
OUT_DIR = 'data/processed/zarr_synthetic_gsb'

TOTAL_HOURS = 1392    
HOURS_TO_FILL = 1  
CHUNK_HOURS = 24
CC_LEN = 500   


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_pairs_and_distances(receivers_path):
    receivers = json.load(open(receivers_path))
    ids = sorted(receivers.keys())

    pairs = []
    for pair_idx, (a, b) in enumerate(itertools.combinations(ids, 2)):
        d = haversine(
            receivers[a]['latitude'], receivers[a]['longitude'],
            receivers[b]['latitude'], receivers[b]['longitude'],
        )
        pairs.append({
            'pair_index': pair_idx,
            'sensor_a': a,
            'sensor_b': b,
            'distance_m': d,
        })
    return pairs


def main():
    pairs = build_pairs_and_distances(RECEIVERS)
    print(pairs)
    distances = np.array([p['distance_m'] for p in pairs])
    print(f'{len(pairs)} pairs | distance {distances.min():.0f}-{distances.max():.0f} m')

    model = dam_model()
#     model = [
#  [Thickness,  Vp,     Vs,     Density],
#  [Thickness,  Vp,     Vs,     Density],
#  [Thickness,  Vp,     Vs,     Density],
#  [Thickness,  Vp,     Vs,     Density]
# ]
    freqs, phase_vels = true_dispersion(model)
    group_vels = true_group_velocity(model, freqs)

    print(f'dispersion: {freqs.min():.1f}-{freqs.max():.1f} Hz | '
          f'phase {phase_vels.min():.0f}-{phase_vels.max():.0f} m/s | '
          f'group {group_vels.min():.0f}-{group_vels.max():.0f} m/s')

    lag_window_s = (N_SAMPLES // 2) / FS
    max_arrival = distances / group_vels.min()
    n_fit = int(np.sum(max_arrival <= lag_window_s))
    
    
    if n_fit < len(pairs):
        print(f'  -> {len(pairs) - n_fit} pairs are too far apart for CC_LEN={CC_LEN}')

    np.save('data/processed/true_dispersion.npy',
            np.column_stack([freqs, phase_vels, group_vels]))

    skipped = 0
    for p in pairs:
        if p['distance_m'] < 1:
            skipped += 1
            continue

        store = PairZarrStore(
            base_dir=OUT_DIR,
            pair_index=p['pair_index'],
            sensor_a_id=p['sensor_a'],
            sensor_b_id=p['sensor_b'],
            total_hours=TOTAL_HOURS,
            cc_len=CC_LEN,
            chunk_hours=CHUNK_HOURS,
        )
        store.save_metadata(p['sensor_a'], p['sensor_b'])

        clean = synth_ccf(freqs, phase_vels, group_vels, p['distance_m'])
        assert len(clean) == N_SAMPLES, f'got {len(clean)} samples'

        ccf_data = {'DPZ': clean, 'DP1': clean, 'DP2': clean}

        for t in range(HOURS_TO_FILL):
            store.save_hour(time_index=t, ccf_data=ccf_data)

        if (p['pair_index'] + 1) % 100 == 0:
            print(f"  {p['pair_index'] + 1}/{len(pairs)} pairs written")

    print(f'done -> {OUT_DIR}  ({skipped} pairs skipped, distance ~0)')


if __name__ == '__main__':
    main()