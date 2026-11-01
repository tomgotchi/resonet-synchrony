"""hc-5 loader/QC on a FAKE session with a known answer (format checks, epochs, re-referencing)."""
import numpy as np
import pytest
import scipy.io

from resonet_sync import data_hc5 as H5
from resonet_sync import qc_hc5 as Q

FS, NCH = 1250, 10


def _fake(d, n_s=40, nch_par=NCH):
    d.mkdir(parents=True, exist_ok=True)
    base = d / "fake_MS.001"
    groups = [list(range(0, 5)), list(range(5, 10))]
    (d / "fake_MS.001.xml").write_text(
        "<parameters><acquisitionSystem><nBits>16</nBits><nChannels>%d</nChannels><samplingRate>20000</samplingRate>"
        "</acquisitionSystem><fieldPotentials><lfpSamplingRate>1250</lfpSamplingRate></fieldPotentials>"
        "<anatomicalDescription><channelGroups>%s</channelGroups></anatomicalDescription></parameters>"
        % (NCH, "".join("<group>" + "".join(f"<channel>{c}</channel>" for c in g) + "</group>" for g in groups)))
    rng = np.random.default_rng(0)
    n = n_s * FS
    t = np.arange(n) / FS
    common = 2000 * np.cos(2 * np.pi * 8 * t)                  # shared reference signal
    X = common[:, None] + 300 * rng.standard_normal((n, NCH))
    X.astype(np.int16).tofile(str(base) + ".eeg")
    nlap = np.zeros(n); nlap[2 * FS:5 * FS] = 1; nlap[20 * FS:23 * FS] = 2      # two 3-s wheel runs
    laps = {"NLapCW": nlap, "NLapCCW": np.zeros(n), "startLfpInd": np.array([6 * FS, 25 * FS]),
            "endLfpInd": np.array([12 * FS, 31 * FS]), "dirChoice": np.array([1, 2]), "corrChoice": np.array([1, 0])}
    scipy.io.savemat(str(base) + "_BehavElectrData.mat",
                     {"Par": {"totNch": nch_par, "SamplingFrequency": FS}, "Laps": laps,
                      "Track": {"speed_MMsec": np.full(n, 100.0)}})
    return d


def test_formats_epochs_and_rereference(tmp_path):
    d = _fake(tmp_path / "s")
    base = H5.find_base(d)
    x = H5.parse_xml(str(base) + ".xml")
    assert x["n_channels"] == NCH and x["lfp_fs"] == FS and len(x["groups"]) == 2
    with pytest.raises(ValueError):
        H5.read_eeg(str(base) + ".eeg", NCH + 1)
    b = H5.load_behav(str(base) + "_BehavElectrData.mat")
    assert b["totNch"] == NCH and b["fs"] == FS
    assert [(r["start"], r["end"]) for r in b["wheel_runs"]] == [(2 * FS, 5 * FS), (20 * FS, 23 * FS)]
    assert len(b["maze_laps"]) == 2 and b["maze_laps"][1]["correct"] == 0.0
    rep = Q.qc_session(d)
    assert rep["format_checks"]["xml_vs_Par_totNch"] is True
    plv = rep["theta_plv_within_group"]
    assert plv["common_reference"] > 0.9 and plv["local_bipolar"] < plv["common_reference"] - 0.3
    assert rep["behaviour"]["n_wheel_runs"] == 2 and rep["behaviour"]["n_maze_laps"] == 2
    assert rep["regions"]["source"].startswith("unknown")       # 10 != 129 channels: no documented map applied


def test_channel_count_mismatch_is_reported(tmp_path):
    d = _fake(tmp_path / "m", nch_par=NCH + 2)
    assert Q.qc_session(d)["format_checks"]["xml_vs_Par_totNch"] is False


def test_documented_map_only_for_129():
    reg, src = H5.region_map(129)
    assert reg[0] == "EC" and reg[32] == "CA1_R" and reg[64] == "CA1_L" and reg[104] == "wheel_ttl" and reg[106] == "sync"
    assert H5.region_map(102)[1].startswith("unknown")
