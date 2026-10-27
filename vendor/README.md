# vendor/

Upstream files copied verbatim. **Do not edit** (TASK_PITCH1.md S9). `resonet_sync/` imports the
published encoder, baselines (EEGNet, ShallowConvNet), data loader and `chance_threshold` from here,
and re-implements only the oscillator parameterization (S2) in `resonet_sync/banks.py`.

| File | Source | Source date | Copied | sha256 |
|---|---|---|---|---|
| `resonance_network.py` | Tom's Google Drive `ResonanceNetwork_OFFICIAL/src/resonance_network.py` (Drive id `1JbKB19v_BB9gmpfh7KPMOua_3_Lybdyb`, 35048 bytes, modified 2026-07-09). Bytes taken from the identical-size copy committed in GitHub `tomgotchi/reso-net` @ `91613e4` (2026-07-27). | 2026-07-09 | 2026-10-27 | `2c9a01a8fa699e39729d4c0429b8acbc1e50e80f912ef492768123f24b882036` |
| `hc2_loader.py` | Tom's Google Drive `HPC-GCM_Official/hc2_loader.py` (Drive id `1knHFS2NsWspm17iy3ZKa7bt88fXAOoSe`, 6658 bytes; file content last modified 2026-07-26, uploaded to that folder 2026-10-09). Built-in `_selftest()` passes. | 2026-07-26 | 2026-10-27 | `cb45233d4a95875f21795b27b147ddbb067dffc51a0163655075de80a07a02f7` |

Same bytes as `phasetrack/vendor/` (copied 2026-10-27).
