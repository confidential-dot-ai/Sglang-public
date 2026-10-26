"""Regression cover for the TP sampling-drift detector.

Black-box behavior that must not come back: a rank whose decoded-token
counter disagrees with its peers stays in rotation until a later collective
deadlocks, with no error raised anywhere.
"""

from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

tp_drift_detector = pytest.importorskip(
    "sglang.srt.managers.scheduler_components.tp_drift_detector"
)
TpDriftDetector = tp_drift_detector.TpDriftDetector


def make_detector(per_call_values, exit_after=3, interval=1):
    """Detector whose gather replays a scripted sequence of rank readings."""
    calls = {"n": 0, "exits": 0}

    def on_drift():
        calls["exits"] += 1

    detector = TpDriftDetector.__new__(TpDriftDetector)
    detector.tp_rank = 0
    detector.tp_size = 4
    detector.on_drift = on_drift
    detector.interval = interval
    detector.exit_after = exit_after
    detector.enabled = True
    detector._calls = 0
    detector._consecutive_divergences = 0

    def fake_gather(_counter):
        values = per_call_values[min(calls["n"], len(per_call_values) - 1)]
        calls["n"] += 1
        return list(values)

    detector._gather = fake_gather
    return detector, calls


def test_agreeing_ranks_do_nothing():
    detector, calls = make_detector([[100, 100, 100, 100]])
    for _ in range(10):
        detector.observe(100)
    assert calls["exits"] == 0


def test_persistent_divergence_takes_the_rank_down():
    """The worker-b signature: tp3 one token ahead, and it never re-converges."""
    detector, calls = make_detector([[100, 100, 100, 101]], exit_after=3)

    detector.observe(100)
    assert calls["exits"] == 0, "one reading is not enough to act on"
    detector.observe(100)
    assert calls["exits"] == 0
    detector.observe(100)
    assert calls["exits"] == 1, "persistent drift must take the rank down"


def test_a_transient_disagreement_does_not_kill():
    """One divergent reading followed by agreement must reset the counter."""
    detector, calls = make_detector(
        [
            [100, 100, 100, 101],
            [200, 200, 200, 200],
            [300, 300, 300, 301],
            [400, 400, 400, 400],
        ],
        exit_after=3,
    )
    for _ in range(4):
        detector.observe(0)
    assert calls["exits"] == 0


def test_a_failed_gather_is_not_treated_as_drift():
    """A gather that raises says nothing about agreement; the watchdog owns
    the stuck-forever case, so this must not kill on its own."""
    detector, calls = make_detector([[0, 0, 0, 0]])

    def boom(_counter):
        raise RuntimeError("gloo timed out")

    detector._gather = boom
    for _ in range(10):
        detector.observe(0)
    assert calls["exits"] == 0


def test_interval_throttles_the_check():
    detector, calls = make_detector([[100, 100, 100, 101]], exit_after=1, interval=5)
    for _ in range(4):
        detector.observe(100)
    assert calls["exits"] == 0, "no check before the interval elapses"
    detector.observe(100)
    assert calls["exits"] == 1


def test_a_peer_one_check_behind_is_not_drift(tmp_path):
    """Ranks reach a check at the same step but not the same instant. Reading a
    peer's previous counter while it grows by hundreds of tokens per check (DSpark
    oracle commits) took a healthy decode worker down."""

    def rank(tp_rank):
        detector = TpDriftDetector.__new__(TpDriftDetector)
        detector.tp_rank, detector.tp_size = tp_rank, 2
        detector._dir, detector._calls = tmp_path, 0
        return detector

    tp0, tp1 = rank(0), rank(1)
    tp0._calls = tp1._calls = 1
    tp0._gather(202)
    tp1._calls = 2
    assert tp1._gather(412) is None, "tp0 has not reached check 2 yet"
    tp0._calls = 2
    assert tp0._gather(412) == [412, 412]
    tp0._calls = tp1._calls = 3
    tp0._gather(678)
    assert tp1._gather(679) == [678, 679], "same-check disagreement is still seen"


def test_workers_sharing_tmp_do_not_read_each_others_counters(tmp_path):
    """Two TP2 workers in one container share /tmp and the same group name. Each
    rank compared itself with the other worker's counters and a healthy decode
    worker took itself down mid-load."""
    with patch("tempfile.gettempdir", return_value=str(tmp_path)):
        decode0, decode1 = (
            [
                TpDriftDetector(
                    group_key="tp:0", worker_id=port, tp_rank=rank, tp_size=2
                )
                for rank in (0, 1)
            ]
            for port in (29602, 29603)
        )
    for detector in decode0 + decode1:
        detector._calls = 1
    decode0[0]._gather(1855)
    decode1[0]._gather(125)
    decode1[1]._gather(125)
    assert decode0[1]._gather(1855) == [1855, 1855]


def test_disabled_detector_never_gathers():
    detector, calls = make_detector([[100, 100, 100, 101]], exit_after=1)
    detector.enabled = False
    for _ in range(10):
        detector.observe(100)
    assert calls["exits"] == 0
