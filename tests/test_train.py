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
    res = smoke(["lin:sync", "nl_coupled:sync", "eegnet"], dict(P1), torch.device("cpu"))
    assert set(res) == {"lin:sync", "nl_coupled:sync", "eegnet"}
    ei = res["nl_coupled:sync"]["evoked_induced"]
    assert {"train", "test_descriptive"} <= set(ei) and "cv_bal_acc_induced" in ei["train"]
    assert "evoked_induced" not in res["eegnet"]
    reg = res["nl_coupled:sync"]["regime"]          # board 2026-11-09: descriptive dynamical regime of the bank
    assert reg["max_re_eig"] < 0 and reg["coupling_over_detuning_median"] < 1 and "regime" not in res["lin:sync"]


def test_no_trial_shuffle_decoder_arm():
    from resonet_sync.run import ALL
    assert not any("trial_shuffle" in a for a in ALL)
