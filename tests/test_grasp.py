import numpy as np
import pytest

from skillforge.contracts import ObjectSpec, Pose
from skillforge.skills.grasp import (
    DOWN,
    frame_quat,
    grasp_to_world,
    mat_to_quat,
    plan_grasps,
    quat_angle,
    quat_from_axis_angle,
    quat_mul,
    quat_to_mat,
    select_grasp,
    yaw_of,
)

MAXW = 0.08


def box(hx, hy, hz):
    return ObjectSpec(name="b", shape="box", size=(hx, hy, hz))


def closing_world(g, q_obj):
    return quat_to_mat(grasp_to_world(g, np.zeros(3), q_obj)[1])[:, 1]


def approach_world(g, q_obj):
    return quat_to_mat(grasp_to_world(g, np.zeros(3), q_obj)[1])[:, 2]


# ---------------------------------------------------------------- quaternion helpers


def test_quat_roundtrip_and_down_convention():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert quat_angle(mat_to_quat(quat_to_mat(q)), q) < 1e-6
    np.testing.assert_allclose(frame_quat([0, 0, -1], [0, -1, 0]), DOWN, atol=1e-9)
    np.testing.assert_allclose(quat_to_mat(DOWN)[:, 2], [0, 0, -1], atol=1e-9)


def test_quat_mul_composes_rotations():
    a = quat_from_axis_angle([0, 0, 1], 0.3)
    b = quat_from_axis_angle([0, 0, 1], 0.4)
    assert yaw_of(quat_mul(a, b)) == pytest.approx(0.7)
    np.testing.assert_allclose(quat_to_mat(quat_mul(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-9)


# ---------------------------------------------------------------- box


def test_cube_grasps_are_top_down_sorted_and_fit():
    gs = plan_grasps(box(0.02, 0.02, 0.02), MAXW)
    assert gs
    assert [g.score for g in gs] == sorted((g.score for g in gs), reverse=True)
    assert all(g.width + 0.01 <= MAXW for g in gs)
    best = gs[0]
    np.testing.assert_allclose(approach_world(best, np.array([1.0, 0, 0, 0])), [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(best.pos, [0, 0, 0], atol=1e-9)  # depth 0.02 reaches the centre


def test_box_respects_max_width():
    # 2*0.05 = 0.10 across x does not fit an 0.08 gripper: only grasps closing across y remain.
    gs = plan_grasps(box(0.05, 0.015, 0.02), MAXW)
    assert gs and all(g.width == pytest.approx(0.03) for g in gs)
    for g in gs:
        assert abs(closing_world(g, np.array([1.0, 0, 0, 0]))[1]) == pytest.approx(1.0)
    assert plan_grasps(box(0.05, 0.05, 0.02), MAXW) == []


def test_tall_box_tcp_stops_short_of_centre():
    g = plan_grasps(box(0.02, 0.02, 0.06), MAXW, depth=0.02)[0]
    np.testing.assert_allclose(g.pos, [0, 0, 0.04], atol=1e-9)


def test_box_grasps_follow_object_yaw_in_world():
    q = quat_from_axis_angle([0, 0, 1], 0.5)
    gs = plan_grasps(box(0.05, 0.015, 0.02), MAXW, obj_quat=q)
    pos, quat, _ = select_grasp(gs, np.array([0.5, 0, 0.02]), q, ee_quat=DOWN)
    closing = quat_to_mat(quat)[:, 1]
    obj_y = quat_to_mat(q)[:, 1]
    assert abs(np.dot(closing, obj_y)) == pytest.approx(1.0)  # closes across the narrow side
    np.testing.assert_allclose(quat_to_mat(quat)[:, 2], [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(pos, [0.5, 0, 0.02], atol=1e-9)


def test_box_on_its_side_still_gets_world_top_down_grasps():
    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)  # object z now points along world -y
    gs = plan_grasps(box(0.02, 0.03, 0.01), MAXW, obj_quat=q)
    assert gs
    for g in gs:
        assert np.dot(approach_world(g, q), [0, 0, -1]) > np.cos(np.pi / 4)
    np.testing.assert_allclose(approach_world(gs[0], q), [0, 0, -1], atol=1e-9)


# ---------------------------------------------------------------- cylinder, sphere, mesh


def test_upright_cylinder_top_down_at_several_yaws():
    cyl = ObjectSpec(name="c", shape="cylinder", size=(0.02, 0.05, 0.0))
    gs = plan_grasps(cyl, MAXW, n_yaw=8)
    assert len(gs) == 8  # end-cap grasps only: side grasps are horizontal
    assert all(g.width == pytest.approx(0.04) for g in gs)
    yaws = {round(float(np.arctan2(closing_world(g, np.array([1.0, 0, 0, 0]))[1], closing_world(g, np.array([1.0, 0, 0, 0]))[0])), 3) for g in gs}
    assert len(yaws) == 8
    assert plan_grasps(ObjectSpec(name="c", shape="cylinder", size=(0.05, 0.05, 0.0)), MAXW) == []


def test_lying_cylinder_grasped_across_diameter_from_above():
    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    cyl = ObjectSpec(name="c", shape="cylinder", size=(0.02, 0.1, 0.0))
    gs = plan_grasps(cyl, MAXW, obj_quat=q)
    assert gs and gs[0].width == pytest.approx(0.04)
    np.testing.assert_allclose(approach_world(gs[0], q), [0, 0, -1], atol=1e-6)
    axis_world = quat_to_mat(q)[:, 2]
    assert abs(np.dot(closing_world(gs[0], q), axis_world)) < 1e-6  # closes across, not along


def test_sphere_top_down_and_width_limit():
    s = ObjectSpec(name="s", shape="sphere", size=(0.03, 0.0, 0.0))
    q = quat_from_axis_angle([1, 1, 0], 1.0)  # orientation must not matter
    gs = plan_grasps(s, MAXW, obj_quat=q)
    assert gs and all(g.width == pytest.approx(0.06) for g in gs)
    for g in gs:
        np.testing.assert_allclose(approach_world(g, q), [0, 0, -1], atol=1e-9)
    assert plan_grasps(ObjectSpec(name="s", shape="sphere", size=(0.04, 0, 0)), MAXW) == []


def test_mesh_returns_no_grasps_yet():
    m = ObjectSpec(name="m", shape="mesh", size=(1, 1, 1), mesh="x.obj", pose=Pose())
    assert plan_grasps(m, MAXW) == []


# ---------------------------------------------------------------- selection


def test_select_prefers_least_wrist_rotation_and_rank_cycles_distinct_grasps():
    gs = plan_grasps(box(0.02, 0.02, 0.02), MAXW)
    q0 = np.array([1.0, 0, 0, 0])
    _, quat0, _ = select_grasp(gs, np.zeros(3), q0, ee_quat=DOWN, rank=0)
    assert quat_angle(quat0, DOWN) < 1e-6
    _, quat1, _ = select_grasp(gs, np.zeros(3), q0, ee_quat=DOWN, rank=1)
    assert abs(np.dot(quat_to_mat(quat1)[:, 1], quat_to_mat(quat0)[:, 1])) < 1e-6  # the other face pair
    _, quat2, _ = select_grasp(gs, np.zeros(3), q0, ee_quat=DOWN, rank=2)
    assert quat_angle(quat2, quat0) < 1e-6  # two distinct top-down grasps on a cube, then wrap
    assert select_grasp([], np.zeros(3), q0) is None
