"""T4: failure tags from the event stream (T1's events: grasp_slip, collision, ik_failure)."""

import pytest

from skillforge.eval.harness import _classify


@pytest.mark.parametrize(
    ("events", "terminated", "timed_out", "tag"),
    [
        ([], False, True, "timeout"),
        ([], False, False, "other"),
        (["grasp_miss"], False, True, "grasp_miss"),
        (["grasp_miss", "grasp:cube"], False, True, "timeout"),  # recovered the grasp, then stalled holding
        (["grasp:cube", "release:cube"], False, True, "wrong_placement"),
        (["grasp:cube", "release:cube", "grasp:cube"], False, True, "timeout"),
        ([], True, False, "object_dropped"),
        (["grasp:cube", "grasp_slip:cube"], True, False, "grasp_slip"),  # the slip explains the drop
        (["grasp:cube", "grasp_slip"], False, True, "grasp_slip"),
        (["collision:arm-table", "grasp_slip:cube"], True, False, "collision"),
        (["grasp_slip", "collision:arm-table", "ik_failure:unreachable"], False, True, "ik_failure"),
        (["ik_failure"], False, True, "ik_failure"),
        (["collisions_checked"], False, True, "timeout"),
        (["sim_unstable"], True, False, "sim_artifact"),
        (["grasp:cube", "grasp_slip", "object_ejected:cube"], True, False, "sim_artifact"),
        (["ik_failure", "collision:hand-table", "object_ejected"], False, True, "sim_artifact"),
        (["grasp:cube", "sim_unstable"], True, False, "sim_artifact"),
        (["object_ejected_check"], False, True, "timeout"),  # name match, not prefix of a longer name  # prefix match is on the event name, not a substring
    ],
)
def test_classify(events, terminated, timed_out, tag):
    got, why = _classify(events, terminated, timed_out)
    assert got == tag
    if tag in ("ik_failure", "collision", "grasp_slip"):
        assert why.startswith(tag)
    if tag == "sim_artifact":
        assert why.startswith(("sim_unstable", "object_ejected"))
