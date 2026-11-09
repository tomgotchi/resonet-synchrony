# Coupling vs rhythmic common input: why passive phase statistics cannot separate them

Review board, 2026-11-09 (Zhao, Volkov; judge-verified). This note gives the formal reason for retiring the
within-session coupling rule after screen 11 (DEVIATION 21). Numerical check: `python scripts/nonident_check.py`
(all three constructions agree to about 1e-15).

## Claim
For linear, Gaussian, stationary dynamics observed passively, **there exist** uncoupled networks driven by latent
rhythmic common inputs whose **cross-spectral matrices S(omega) equal those of a directly coupled network at every
frequency** (an existence result: the two classes overlap; not every common-input model matches every coupled one). For Gaussian stationary processes S(omega) fixes the joint law. So **every** statistic of the
observations has the same distribution under both models: coherence, PLV, phase lag, Granger causality and
Gaussian transfer entropy in either direction, LDS likelihoods, and any decoder's accuracy. Screens 1-11 measured
how far the rule's extra assumptions carried it; this note says why no rule of that kind can be valid in this regime:
the model class contains exact equivalents.

## Construction 1: symmetric diffusive coupling (Volkov)
Two identical units, complex OU form, observed through their real parts:

  dz_j = (lambda + i omega_0) z_j dt + k (z_l - z_j) dt + sigma dW_j,   lambda < 0.

This equals in law two **uncoupled** units with pole lambda + i omega_0 - 2k and private noise sigma, both driven by
one common Lorentzian rhythm s (pole lambda + i omega_0, unit noise) with gain c, where

  c^2 = 2 sigma^2 k (k + |lambda|).

Proof: in the eigenbasis (1, +-1)/sqrt(2) the coupled pair is a sum mode (pole lambda + i omega_0) and a difference
mode (pole lambda + i omega_0 - 2k). The common rhythm reproduces the sum mode's excess power, at every
frequency, because the two poles share omega_0 (the difference of squared denominators is constant in omega).

**N units** on any symmetric weighted graph with Laplacian L (eigenpairs mu_m, v_m):
dz = ((lambda + i omega_0) I - kappa L) z dt + sigma dW equals uncoupled units with pole
lambda + i omega_0 - kappa mu_max plus **N - 1 common rhythms**. Rhythm m has pole lambda + i omega_0 - kappa mu_m,
spatial loading v_m and gain

  c_m^2 = sigma^2 kappa (mu_max - mu_m) (kappa (mu_max + mu_m) + 2 |lambda|).

The common-input model lies in the same parametric family as the coupled one. **Construction 1 needs identical units
(same lambda and omega_0) and symmetric coupling with real modal loadings**; with detuned units it is approximate
(relative S error 1.6e-2 at 0.5 Hz, 5.7e-2 at 1 Hz detuning; Volkov, board round 3).

**What this does and does not say about screen 11.** Screen 11's units were detuned (8 +/- 1 Hz), had a cubic term,
and its rhythmic-common-input cells used one 2 Hz-wide rhythm with complex (15 deg gradient) loadings, matched on PLV
rather than on S(omega). So its R cells were **not** exact equivalents of its coupled cells, and this note does not
prove that no statistic could have separated them. The retirement of the rule rests on its pre-committed criterion
(P(fire) 0.94-1.00 in those cells; DEVIATION 21). What this note adds is that the model class contains exact
equivalents, so no passive rule of that kind can be valid in general. (Correction, board 2026-11-09 round 3.)

## Construction 2: directed coupling (Zhao)
Model A: an AR(2) theta unit 1 drives unit 2 with gain b; both observed with noise. Model B: no unit-to-unit
coupling. A latent rhythm L appears in unit 1 together with a private, non-propagating share alpha of unit 1's
rhythm, and drives unit 2 with gain b / (1 - alpha). Unit 2 receives a compensating private spectrum. The two
spectral matrices are identical for every alpha up to alpha_max (0.66 at a theta coherence of 0.31). This is
exact whenever unit 2 has strictly positive private noise. At alpha = 0, model B **is** model A: what cannot be
identified is the **share of unit 1's rhythm that propagates** to unit 2. The construction weakens as coherence
rises (alpha_max is only 0.002 at coherence 0.97). Model B's private spectrum is not of
"AR(2) plus white" form. Identification is therefore possible only by assuming a shape for the noise spectrum,
which is the assumption that failed under 1/f and spatially correlated noise in screen 11 (cell Q1).

L is **not observed** in model B: unit 1 is L plus a private rhythm plus noise. The construction is general for
any minimum-phase path L -> unit 1.

## Assumptions (all needed)
1. Linear dynamics. For the nl_coupled bank and the screen generators, report the cubic-to-linear damping ratio.
   Construction 1 additionally needs identical units (same lambda, omega_0) and symmetric coupling.
2. Gaussian, circular white driving noise.
3. Stationarity, with stable poles (lambda < 0).
4. Strictly positive private noise, so S(omega) is positive definite.
5. Passive observation; the drivers are not observed.
6. No limit on the number of latent drivers.

## What breaks these constructions
These conditions break the **linear-Gaussian constructions above**. They have **not** been shown to identify coupling:
identifiability against nonlinear latent drivers is open. A self-sustained common driver can also produce
arcsin-type (Adler) locking with phase slips in each unit, and an amplitude-weighted phase pull correlates with a
latent driver's amplitude. (Correction, board round 3.)
- **Non-Gaussian dynamics:**
  - self-sustained units (lambda > 0, limit cycles);
  - amplitude-dependent phase pull: under coupling, phase pull scales with the partner's amplitude (r_l / r_j),
    which a common drive does not reproduce;
  - non-isochronicity (frequency depends on amplitude);
  - arcsin-type locking lags with phase slips (coupling), against arctan lags of driven filters.
- **An observed driver.** For example an EC recording: partial coherence given the driver. A proxy of L helps;
  two proxies give full identification (Miao et al. 2018).
- **Independent non-Gaussian sources** (Hoyer et al. 2008).
- **A limit on the number of drivers.** A single septal rhythm gives a rank-1 common part. Tetrad constraints with
  four or more sites test a single-driver hypothesis; a chain violates S12 S34 = S13 S24. Lead-field mixing also
  lowers rank, which weakens this route for LFP.
- **Perturbations, or a known switching time.**
- **Known intrinsic unit parameters.** The equivalent uncoupled units carry extra self-damping kappa mu_max; if the
  units' own damping is known independently, construction 1 is broken.

## Consequences for this project
- The TE/LDS rule is retired on formal grounds, not only because screen 11's rhythmic-common-input cells fired
  it (P(fire) 0.94-1.00).
- Any future coupling claim from passive data must name its identifying assumption (non-Gaussianity, an
  observed driver, a rank limit or a perturbation) and test it.
- The hc-5 plan asks a different question that is identifiable in principle: which **generators** (CA3 radiatum
  vs EC3 lacunosum-moleculare) carry theta, through their CSD laminar signatures. It does not ask whether two
  sites are coupled.
- Phase 2's emergent data leave **no headroom** for non-Gaussian information: a Whittle (purely spectral) classifier
  scores 0.999 over 10 seeds (`results/phase2/whittle_oracle.json`). Second-order information saturates accuracy,
  so the "nl vs lin" decoder contrast could not show a non-Gaussian advantage there. (This does not show that the
  generator itself is linear-Gaussian.)

## Not covered
- Finite-sample power: how quickly the non-Gaussian signatures above become detectable.
- The supercritical (limit-cycle) regime. A cheap oracle-only check, Whittle plus one amplitude-weighted
  phase-pull statistic on Stuart-Landau pairs, is on the board's plan.
- The lead-field problem: LFP and EEG sites are mixtures of generators, so "units" are not observed directly.
