import numpy as np

from resonet_sync.synth_smoke import tiny_dataset
from resonet_sync.train import Split


def test_partners_same_class_different_trial_same_split():
    X, y, _, _ = tiny_dataset(n_per_class_tr=5)
    s = Split(X, y, 0)
    idx = np.arange(len(y))
    p = s.sample_partners(idx)
    assert np.all(y[p] == y[idx]) and np.all(p != idx)
    assert np.all(s.fixed == Split(X, y, 0).fixed)   # deterministic eval partners


def test_smoke_end_to_end():
    import torch
    from resonet_sync.run import P1, smoke
    res = smoke(["lin:sync", "nl_coupled:trial_shuffle", "eegnet"], dict(P1), torch.device("cpu"))
    assert set(res) == {"lin:sync", "nl_coupled:trial_shuffle", "eegnet"}
