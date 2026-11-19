import sys, numpy as np
sys.path.insert(0,'/home/user/resonet-synchrony/pairs_trader')
from pairs_trader.coint import symmetric_eg
rng=np.random.default_rng(5); n=504
for bounce in (0.0, 2e-4):
  for name in ("clone_null","null"):
    rej=0; R=400
    for _ in range(R):
        lb=np.cumsum(0.018*rng.standard_normal(n))
        if name=="clone_null":
            sig=np.exp(rng.uniform(np.log(1e-4),np.log(4e-4)))
            la=lb*rng.uniform(.97,1.03)+np.cumsum(sig*rng.standard_normal(n))
        else:
            la=np.cumsum(0.018*rng.standard_normal(n))
        la=la+bounce*rng.standard_normal(n); lb2=lb+bounce*rng.standard_normal(n)
        rej+=symmetric_eg(la,lb2,1)['p']<0.05
    print(f"bounce {bounce*1e4:.0f}bp {name:10s} EG rejection {rej/R:.3f}")
