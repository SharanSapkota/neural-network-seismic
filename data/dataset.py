
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.make_pairs import load_pairs_with_distance, build_pairs
from data.noise_injector import add_noise

class CCFDataset(Dataset):
    def __init__(self, noisy, clean, levels):
        # add channel dim: (N, 1001) -> (N, 1, 1001)
        self.noisy = torch.from_numpy(noisy).float().unsqueeze(1)
        self.clean = torch.from_numpy(clean).float().unsqueeze(1)
        self.levels = torch.from_numpy(levels).float()

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, i):
        return self.noisy[i], self.clean[i], self.levels[i]

def split_arrays(noisy, clean, levels, pairs, all_distance, seed=0, train_frac=0.8, val_frac=0.1):
    n = len(levels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) # Random suffeling of levels

    train_end = int(n * train_frac)
    validation_end = int(n * (train_frac + val_frac))
    training_data, validation_data, te = idx[:train_end], idx[train_end:validation_end], idx[validation_end:]

    def pick(sel):
        return (noisy[sel], clean[sel], levels[sel],
                [pairs[i] for i in sel],
                all_distance[sel])

    return pick(training_data), pick(validation_data), pick(te)


def make_loaders(receivers_path, rng, batch_size=32, n_noise_per_ccf=5, noise_fn=None, seed=0):

    if noise_fn is None:
        noise_fn = add_noise

    pairs = load_pairs_with_distance(receivers_path)          # (a, b, dist) list
    print(pairs)
    noisy, clean, levels, pair_keys, all_distance = build_pairs(     # 5 returns now
        pairs, rng, n_noise_per_ccf=n_noise_per_ccf, noise_fn=noise_fn
    )
    print(f"built {len(levels)} examples from {len(pairs)} pairs")

    train, val, test = split_arrays(                          # pass all 5
        noisy, clean, levels, pair_keys, all_distance, seed=seed
    )

    train_loader = DataLoader(CCFDataset(*train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CCFDataset(*val),   batch_size=batch_size)
    test_loader = DataLoader(CCFDataset(*test),  batch_size=batch_size)

    return train_loader, val_loader, test_loader