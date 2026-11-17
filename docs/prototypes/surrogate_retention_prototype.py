# Board 2026-11-17 prototype (judge): retained cross-structure of the published surrogate, synthetic input, checkpoint 1.
# Not a result on real data. The implementing model turns this into scripts/surrogate_retention.py (docs/FIX_BRIEF.md, 1d).
import sys, itertools, numpy as np, torch
sys.path.insert(0,'.')
torch.set_num_threads(1)
from resonet_sync._vendor import rn
from scipy import signal
R=rn()
sd=torch.load('/home/user/reso-net/results_hg/RNsync_S1.pt',map_location='cpu',weights_only=True)
m=R.ResonanceNetwork(n_channels=128,n_osc=16,n_classes=4,readout='sync'); print(m.load_state_dict(sd,strict=False)); m.eval()
rng=np.random.default_rng(0)
B,T,C=6,1125,128
# 1/f-like multichannel input with a shared-source mixing, 4 Hz high-pass, z-scored (like the cache after decim 2)
src=np.cumsum(rng.normal(size=(B,6,T+400)),axis=-1)*0+0
f=np.fft.rfftfreq(T+400,1/250); spec=(rng.normal(size=(B,6,len(f)))+1j*rng.normal(size=(B,6,len(f))))/np.maximum(f,1)[None,None]**0.9
src=np.fft.irfft(spec,n=T+400,axis=-1)[...,200:-200]
X=np.einsum('cs,bst->bct',rng.normal(size=(C,6)),src)+0.3*rng.normal(size=(B,C,T))
h=signal.firwin(201,4.0,pass_zero=False,fs=250); X=signal.filtfilt(h,[1.0],X,axis=-1)
X=((X-X.mean(axis=(0,2),keepdims=True))/X.std(axis=(0,2),keepdims=True)).astype(np.float32)
with torch.no_grad():
    logits,traj=m(torch.tensor(X),return_traj=True)
z=traj; u=z/(z.abs()+1e-6)                       # (B,T-1,N)
Tm=u.shape[1]; sh=(m.surr_frac*Tm).long()
us=torch.stack([torch.roll(u[:,:,i],int(sh[i]),dims=1) for i in range(16)],dim=2)
def C(U): return torch.einsum('bti,btj->bij',U,U.conj())/Tm
C0,C1=C(u),C(us)
iu,ju=torch.triu_indices(16,16,1)
a0=C0[:,iu,ju].abs().mean(0).numpy(); a1=C1[:,iu,ju].abs().mean(0).numpy()
d=np.array([min(abs(int(sh[i])-int(sh[j])),Tm-abs(int(sh[i])-int(sh[j])))/250 for i,j in zip(iu.tolist(),ju.tolist())])
print('mean |C| over pairs, unshifted %.3f, shifted %.3f (ratio %.2f)'%(a0.mean(),a1.mean(),a1.mean()/a0.mean()))
for lo,hi in ((0,0.25),(0.25,0.5),(0.5,1.0),(1.0,2.5)):
    k=(d>=lo)&(d<hi); print(f'  relative shift {lo:4.2f}-{hi:4.2f} s: {k.sum():3d} pairs, |C| shifted/unshifted = {a1[k].mean()/a0[k].mean():.2f}')
# share of the squared cross-structure that survives: sum |C1|^2 / sum |C0|^2
print('squared cross-structure retained: %.0f%%'%(100*(a1**2).sum()/(a0**2).sum()))
# decomposition: C = <u_i><u_j>* (shift-invariant product of mean phasors) + covariance part
mu_=u.mean(1)                                          # (B,N)
D=torch.einsum('bi,bj->bij',mu_,mu_.conj())
cov0=C0-D; cov1=C1-D
rms=lambda M: float(torch.sqrt((M[:,iu,ju].abs()**2).mean()))
print('rms |C| unshifted %.3f | shifted %.3f | DC term |<u_i><u_j>*| %.3f | covariance part: unshifted %.3f, shifted %.3f'%(rms(C0),rms(C1),rms(D),rms(cov0),rms(cov1)))
print('mean |<u_i>| (how far the phasors sit from a uniform circle): %.3f'%float(mu_.abs().mean()))
