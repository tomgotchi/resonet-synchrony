"""Training/eval loop. Replicates vendor `train_model` exactly (Adam lr 1e-3, wd 1e-4, cosine LR,
label smoothing 0.05, grad-clip 1.0, online augmentation, early stopping on VALIDATION loss with
patience 15, best-val checkpoint restored) and adds same-class partner sampling for
`trial_shuffle`. No test data is touched during training or model selection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ._vendor import rn


class Split:
    """Tensors for one split plus a same-class partner sampler (partners come from the SAME split)."""

    def __init__(self, X, y, seed):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.by_class = {int(c): np.flatnonzero(np.asarray(y) == c) for c in np.unique(y)}
        self.rng = np.random.default_rng(seed)
        self.fixed = self.sample_partners(np.arange(len(y)))  # deterministic map for eval

    def sample_partners(self, idx):
        out = np.empty(len(idx), dtype=int)
        yy = self.y.numpy()
        for k, i in enumerate(idx):
            pool = self.by_class[int(yy[i])]
            if len(pool) < 2:
                out[k] = i
                continue
            j = i
            while j == i:
                j = pool[self.rng.integers(len(pool))]
            out[k] = j
        return out

    def __len__(self):
        return len(self.y)


def _logits(model, x, xp):
    return model(x, xp) if getattr(model, "needs_partner", False) else model(x)


@torch.no_grad()
def evaluate(model, split, device, crit, batch=128, n_classes=4):
    model.eval()
    preds, loss_sum = [], 0.0
    for s in range(0, len(split), batch):
        idx = np.arange(s, min(s + batch, len(split)))
        x = split.X[idx].to(device)
        xp = split.X[split.fixed[idx]].to(device) if getattr(model, "needs_partner", False) else None
        y = split.y[idx].to(device)
        lg = _logits(model, x, xp)
        loss_sum += crit(lg, y).item() * len(idx)
        preds.append(lg.argmax(1).cpu())
    p = torch.cat(preds).numpy()
    y = split.y.numpy()
    acc = float((p == y).mean())
    recalls = [float((p[y == c] == c).mean()) for c in np.unique(y)]
    return dict(loss=loss_sum / len(split), acc=acc, bal_acc=float(np.mean(recalls)), n=int(len(y)))


def train(model, tr: Split, va: Split, device, *, augment=True, max_epochs=80, lr=1e-3,
          weight_decay=1e-4, patience=15, label_smoothing=0.05, grad_clip=1.0, batch=128, seed=42,
          log=None):
    R = rn()
    torch.manual_seed(seed)
    model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    rng = np.random.default_rng(seed)
    best, best_state, bad = float("inf"), None, 0
    hist = dict(train_loss=[], val_loss=[], val_acc=[])
    needs_p = getattr(model, "needs_partner", False)
    for ep in range(max_epochs):
        model.train()
        order = rng.permutation(len(tr))
        tl = []
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            x = tr.X[idx].to(device)
            xp = tr.X[tr.sample_partners(idx)].to(device) if needs_p else None
            y = tr.y[idx].to(device)
            if augment:
                x = R.augment_batch(x)
                if xp is not None:
                    xp = R.augment_batch(xp)
            opt.zero_grad(set_to_none=True)
            loss = crit(_logits(model, x, xp), y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tl.append(loss.item())
        sched.step()
        v = evaluate(model, va, device, crit, batch)
        hist["train_loss"].append(float(np.mean(tl))); hist["val_loss"].append(v["loss"]); hist["val_acc"].append(v["acc"])
        if log:
            log(f"    ep {ep:3d} train {np.mean(tl):.4f} val {v['loss']:.4f} acc {v['acc']:.3f}")
        if v["loss"] < best - 1e-5:
            best, bad = v["loss"], 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]))
    hist["epochs_run"] = len(hist["val_loss"])
    return model, hist
