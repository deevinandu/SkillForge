"""T6: camera frames, pose estimators, the PerceivedEnv wrapper and the error model."""

import json
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import Action, ObjectState, Observation, SimEnv, StepInfo, TaskSpec
from skillforge.eval import evaluate
from skillforge.perception import (
    CameraFrame,
    DepthEstimator,
    Estimate,
    OracleEstimator,
    PerceivedEnv,
    SimFrameSource,
    backproject,
    make_depth_env,
    make_oracle_env,
    measure,
    overview_camera,
    project,
)
from skillforge.perception.estimators import (
    color_mask,
    fit_box,
    fit_cylinder,
    fit_sphere,
    quat_yaw,
    voxel_downsample,
    wrap,
    yaw_quat,
)
from skillforge.sim import StubEnv
from skillforge.skills import PickPlace

TASKS = Path(__file__).parent.parent / "tasks"


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate(json.loads((TASKS / f"{name}.json").read_text()))


# ---------------------------------------------------------------- geometry, no rendering


def box_cloud(half, centre, yaw, n=4000, seed=0):
    """Points on the top face and the two side faces a camera on the +x, -y side would see."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, 2))
    top = np.column_stack([u[:, 0] * half[0], u[:, 1] * half[1], np.full(n, half[2])])
    side_x = np.column_stack([np.full(n, half[0]), u[:, 0] * half[1], u[:, 1] * half[2]])
    side_y = np.column_stack([u[:, 0] * half[0], np.full(n, -half[1]), u[:, 1] * half[2]])
    local = np.concatenate([top, side_x[: n // 2], side_y[: n // 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ R.T + centre


@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.6, 0.78])
def test_fit_box_recovers_a_cube(yaw):
    half, centre = np.array([0.02, 0.02, 0.02]), np.array([0.5, -0.1, 0.02])
    pos, quat, conf = fit_box(box_cloud(half, centre, yaw), half)
    assert np.linalg.norm(pos - centre) < 0.002
    assert abs(wrap(quat_yaw(quat) - yaw, np.pi / 2)) < np.radians(2)  # a cube repeats every 90 deg
    assert conf > 0.8


def test_fit_box_yaw_of_a_rectangle_follows_its_long_side():
    half, centre = np.array([0.04, 0.015, 0.02]), np.array([0.4, 0.1, 0.02])
    for yaw in (0.2, 1.3, -1.2):
        pos, quat, _ = fit_box(box_cloud(half, centre, yaw), half)
        assert np.linalg.norm(pos - centre) < 0.002
        assert abs(wrap(quat_yaw(quat) - yaw, np.pi)) < np.radians(2)


def test_fit_box_reports_the_yaw_nearest_the_previous_one():
    half, centre = np.array([0.02, 0.02, 0.02]), np.zeros(3)
    _, quat, _ = fit_box(box_cloud(half, centre, 0.1), half, prev_yaw=np.pi / 2 + 0.05)
    assert abs(quat_yaw(quat) - (np.pi / 2 + 0.1)) < np.radians(2)


def test_fit_box_confidence_drops_when_half_the_top_is_hidden():
    half, centre = np.array([0.02, 0.02, 0.02]), np.zeros(3)
    pts = box_cloud(half, centre, 0.0)
    _, _, conf = fit_box(pts[pts[:, 0] < -0.005], half)
    assert conf < 0.3


def test_fit_cylinder_and_sphere():
    rng = np.random.default_rng(1)
    r, hh, centre = 0.02, 0.03, np.array([0.5, 0.1, 0.03])
    a, rad = rng.uniform(0, 2 * np.pi, 3000), 0.02 * np.sqrt(rng.uniform(0, 1, 3000))
    top = np.column_stack([rad * np.cos(a), rad * np.sin(a), np.full(3000, hh)])
    a2 = rng.uniform(-0.5, 1.5, 1500)  # the part of the wall facing the camera
    wall = np.column_stack([r * np.cos(a2), r * np.sin(a2), rng.uniform(-hh, hh, 1500)])
    fit = fit_cylinder(np.concatenate([top, wall]) + centre, r, hh)
    assert np.linalg.norm(fit.pos - centre) < 0.002 and fit.confidence > 0.8 and not fit.lying
    # with the top hidden (fingers over it) the side wall alone still gives the horizontal centre
    hidden = fit_cylinder(wall[wall[:, 2] < 0.0] + centre, r, hh)
    assert hidden.confidence < 0.7 and hidden.xy_confidence > 0.9
    assert np.linalg.norm(hidden.xy - centre[:2]) < 0.001

    d = rng.normal(size=(3000, 3))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    d = d[(d[:, 2] > -0.2) & (d[:, 0] > -0.3)]  # the cap a camera above and in front would see
    c = np.array([0.45, -0.2, 0.025])
    pos, _, conf = fit_sphere(d * 0.025 + c, 0.025)
    assert np.linalg.norm(pos - c) < 0.001 and conf > 0.9


@pytest.mark.parametrize("yaw", [0.0, 0.7, 2.0])
def test_fit_cylinder_on_its_side(yaw):
    """A can lying on the table, seen from one side: the visible half of its wall and one cap."""
    rng = np.random.default_rng(2)
    r, hh = 0.02, 0.04
    ang = rng.uniform(-0.4, np.pi - 0.6, 4000)  # the upper wall and the side facing the camera
    wall = np.column_stack([rng.uniform(-hh, hh, 4000), r * np.cos(ang), r * np.sin(ang)])
    a, rad = rng.uniform(0, 2 * np.pi, 800), r * np.sqrt(rng.uniform(0, 1, 800))
    cap = np.column_stack([np.full(800, hh), rad * np.cos(a), rad * np.sin(a)])
    local = np.concatenate([wall, cap])  # axis along local x
    c, s = np.cos(yaw), np.sin(yaw)
    centre = np.array([0.5, -0.2, r])
    pts = local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + centre
    fit = fit_cylinder(pts, r, hh)
    assert fit.lying and fit.confidence > 0.8
    assert np.linalg.norm(fit.pos - centre) < 0.002
    w, x, y, z = fit.quat  # the object's z axis (third column of its rotation) lies along the can
    axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    assert abs(abs(axis @ np.array([c, s, 0.0])) - 1) < 0.01


def test_voxel_downsample_is_even_and_deterministic():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.uniform(0, 0.01, (5000, 3)), rng.uniform(0.02, 0.03, (50, 3))])
    out = voxel_downsample(pts, 0.002)
    assert len(out) < 250 and np.array_equal(out, voxel_downsample(pts, 0.002))
    assert np.all(out.min(axis=0) >= pts.min(axis=0)) and np.all(out.max(axis=0) <= pts.max(axis=0))


def test_color_mask_picks_the_hue_not_the_brightness():
    img = np.zeros((2, 3, 3), dtype=np.uint8)
    img[0, 0], img[0, 1], img[0, 2] = (217, 38, 38), (110, 20, 20), (38, 90, 217)  # red, dark red, blue
    img[1, 0], img[1, 1] = (158, 128, 97), (200, 200, 200)  # table brown, grey
    m = color_mask(img, (0.85, 0.15, 0.15))
    assert m.tolist() == [[True, True, False], [False, False, False]]


def synthetic_frame():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])  # camera looking straight down
    T[:3, 3] = [0.5, 0.0, 1.0]
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    return CameraFrame(rgb=np.zeros((480, 640, 3), np.uint8), depth=np.full((480, 640), 1.0, np.float32), K=K, T_world_cam=T)


def test_backproject_and_project_are_inverses():
    f = synthetic_frame()
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:110, 200:220] = True
    pts = backproject(f, mask)
    assert pts.shape == (200, 3) and np.allclose(pts[:, 2], 0.0)  # 1 m below a camera at z = 1
    uv, z = project(f, pts)
    v, u = np.nonzero(mask)
    assert np.allclose(uv, np.column_stack([u + 0.5, v + 0.5])) and np.allclose(z, 1.0)
    f.depth[105, 205] = 0.0  # no return
    assert len(backproject(f, mask)) == 199


# ---------------------------------------------------------------- PerceivedEnv on the stub env


class ScriptedEstimator:
    """found/not found per call from a script; when found, reports the env's pose plus `offset`."""

    needs_frames = False

    def __init__(self, script, offset=(0.0, 0.0, 0.0)):
        self.script, self.offset, self.calls = list(script), np.array(offset), 0

    def reset(self, task):
        self.calls = 0

    def estimate(self, frames, obs):
        found = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return {
            n: Estimate(pos=o.pos + self.offset, quat=o.quat.copy(), found=found, confidence=float(found))
            for n, o in obs.objects.items()
        }


def make_stub_oracle(task):
    return PerceivedEnv(StubEnv(task), OracleEstimator())


def test_perceived_env_satisfies_the_contract(stub_task):
    assert isinstance(make_stub_oracle(stub_task), SimEnv)


def test_oracle_wrapper_changes_nothing(stub_task):
    a, eps_a = evaluate(StubEnv, PickPlace(), stub_task, range(10))
    b, eps_b = evaluate(make_stub_oracle, PickPlace(), stub_task, range(10))
    drop = {"duration_s"}
    assert [e.model_dump(exclude=drop) for e in eps_a] == [e.model_dump(exclude=drop) for e in eps_b]
    assert a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})


def test_same_seed_same_episode_with_estimator_noise(stub_task):
    def run():
        env = PerceivedEnv(StubEnv(stub_task), OracleEstimator(noise_std=0.003))
        obs = env.reset(7)
        out = [obs.objects["cube"].pos.copy()]
        for _ in range(5):
            obs, _ = env.step(Action(ee_pos=np.array([0.4, 0.0, 0.3])))
            out.append(obs.objects["cube"].pos.copy())
        return np.array(out)

    a, b = run(), run()
    assert np.array_equal(a, b) and np.ptp(a, axis=0).max() > 0  # noisy, and reproducibly so


def test_success_comes_from_the_inner_env_not_the_estimate(stub_task):
    """An estimator that is 30 cm off cannot make a failed episode pass or a passed one fail."""
    bin_centre = np.array(stub_task.scene.region("bin").center)
    cube = np.array(stub_task.scene.object("cube").pose.pos)
    env = PerceivedEnv(StubEnv(stub_task), ScriptedEstimator([True], offset=bin_centre - cube), in_hand=False)
    obs = env.reset(0)
    obs, info = env.step(Action(ee_pos=np.array([0.4, 0.0, 0.3])))
    assert not info.success  # the estimate sits in the bin; the cube does not

    report, _ = evaluate(make_stub_oracle, PickPlace(), stub_task, range(5))
    assert report.success_rate == 1.0


def test_unseen_object_keeps_last_estimate_and_reports_once_per_streak(stub_task):
    env = PerceivedEnv(StubEnv(stub_task), ScriptedEstimator([True, False, False, True, False]), in_hand=False)
    obs = env.reset(3)
    seen_at = obs.objects["cube"].pos.copy()
    hover = Action(ee_pos=np.array([0.4, 0.0, 0.3]))
    events, held = [], []
    for _ in range(4):
        obs, info = env.step(hover)
        events.append(info.events)
        held.append(np.array_equal(obs.objects["cube"].pos, seen_at))
        if not info.events and not held[-1]:
            seen_at = obs.objects["cube"].pos.copy()  # a fresh sighting (the stub's obs are noisy)
    assert events == [["perception_miss:cube"], [], [], ["perception_miss:cube"]]
    assert held == [True, True, False, True]  # held while unseen, updated when seen again


def test_never_seen_object_starts_at_its_nominal_pose(stub_task):
    env = PerceivedEnv(StubEnv(stub_task), ScriptedEstimator([False]), in_hand=False)
    obs = env.reset(3)
    assert np.allclose(obs.objects["cube"].pos, stub_task.scene.object("cube").pose.pos)


def test_estimate_hz_runs_the_estimator_less_often(stub_task):
    est = ScriptedEstimator([True])
    env = PerceivedEnv(StubEnv(stub_task), est, estimate_hz=5.0)  # control is 20 Hz
    env.reset(0)
    for _ in range(8):
        env.step(Action(ee_pos=np.array([0.4, 0.0, 0.3])))
    assert est.calls == 3  # steps 0, 4, 8 of 9 observations


class CarryEnv:
    """A scripted inner env: the gripper closes on the object at step 2 and then moves in +x."""

    def __init__(self, task):
        self.task, self.k = task, 0

    def _obs(self):
        ee = np.array([0.5 + 0.01 * max(0, self.k - 2), -0.1, 0.02])
        width = 0.08 if self.k < 2 else 0.036  # measured opening: stays > 0 around a 4 cm cube
        return Observation(
            t=self.k * 0.05, qpos=np.zeros(7), qvel=np.zeros(7), ee_pos=ee, ee_quat=np.array([0.0, 1, 0, 0]),
            gripper_width=width, objects={"cube": ObjectState(pos=ee.copy(), quat=np.array([1.0, 0, 0, 0]))},
        )

    def reset(self, seed):
        self.k = 0
        return self._obs()

    def step(self, action):
        self.k += 1
        return self._obs(), StepInfo(success=False, terminated=False)

    def randomization_sample(self):
        return {}

    def render(self, camera=None):
        return None

    def close(self):
        pass


@pytest.mark.parametrize("in_hand", [True, False])
def test_hidden_object_in_the_gripper_is_carried_with_the_end_effector(stub_task, in_hand):
    script = [True, True, True] + [False] * 10  # the fingers hide the cube once they close
    env = PerceivedEnv(CarryEnv(stub_task), ScriptedEstimator(script), in_hand=in_hand)
    obs = env.reset(0)
    events = []
    for _ in range(8):
        obs, info = env.step(Action(ee_pos=np.zeros(3)))
        events += info.events
    moved = obs.objects["cube"].pos[0] - 0.5
    if in_hand:
        assert abs(moved - 0.06) < 1e-9 and events == []  # carried 6 cm with the tcp, no miss
    else:
        assert moved == 0.0 and events == ["perception_miss:cube"]


class WallEstimator:
    """Like a cylinder whose top the fingers hide: no confident pose, but a good side-wall xy."""

    needs_frames = False
    min_confidence = 0.7

    def __init__(self, slide):
        self.slide = np.array(slide)

    def reset(self, task):
        self.calls = 0

    def estimate(self, frames, obs):
        self.calls += 1
        o = obs.objects["cube"]
        if self.calls <= 3:  # clearly visible before the grasp
            return {"cube": Estimate(pos=o.pos.copy(), quat=o.quat.copy(), found=True, confidence=1.0)}
        xy = o.pos[:2] + self.slide  # where the object really is: it slid in the fingers
        return {"cube": Estimate(pos=o.pos + 0.01, quat=o.quat.copy(), found=True, confidence=0.1, xy=xy, xy_confidence=0.95)}


def test_in_hand_carry_takes_the_horizontal_position_from_a_confident_wall_fit(stub_task):
    env = PerceivedEnv(CarryEnv(stub_task), WallEstimator(slide=(0.0, 0.015)))
    obs = env.reset(0)
    for _ in range(8):
        obs, _ = env.step(Action(ee_pos=np.zeros(3)))
    pos = obs.objects["cube"].pos
    assert np.allclose(pos[:2], [0.56, -0.1 + 0.015])  # xy follows the wall fit, slide included
    assert np.isclose(pos[2], 0.02)  # z is carried, not taken from the coarse estimate


class CoarseEstimator:
    """Always a coarse (low-confidence) estimate, drifting 0.5 mm further from the env's pose per call."""

    needs_frames = False
    min_confidence = 0.7

    def reset(self, task):
        self.calls = 0

    def estimate(self, frames, obs):
        self.calls += 1
        o = obs.objects["cube"]
        pos = o.pos + [0.0005 * self.calls, 0.0, 0.0]
        return {"cube": Estimate(pos=pos, quat=o.quat.copy(), found=True, confidence=0.1)}


@pytest.mark.parametrize("ee, follows", [((0.5, -0.1, 0.30), True), ((0.5, -0.1, 0.08), False)])
def test_coarse_estimate_is_used_only_when_the_gripper_is_clear_of_the_object(stub_task, ee, follows):
    """Far from the gripper the fingers cannot be what hides the object, so a coarse estimate is
    news about the object; next to the gripper it is occlusion, and the last pose stands."""
    clean = stub_task.model_copy(update={"scene": stub_task.scene.model_copy(update={"randomization": type(stub_task.scene.randomization)()})})
    est = CoarseEstimator()
    env = PerceivedEnv(StubEnv(clean), est)
    env.reset(0)
    for _ in range(40):
        obs, _ = env.step(Action(ee_pos=np.array(ee)))
    off = obs.objects["cube"].pos[0] - clean.scene.object("cube").pose.pos[0]
    latest = 0.0005 * est.calls
    if follows:
        assert np.isclose(off, latest)
    else:  # frozen at the value it had when the gripper came within clear_radius
        assert 0 < off < latest - 0.005


# ---------------------------------------------------------------- MuJoCo, no rendering


def test_error_model_recovers_the_envs_own_observation_noise():
    """Through the oracle the only error is the env's Gaussian obs noise, whose std is known."""
    pytest.importorskip("mujoco")
    task = load("franka_pick_cube_to_bin_hard")
    rep = measure(make_oracle_env, PickPlace(), task, range(4))
    rnd = task.scene.randomization.obs_noise_std
    assert rep.n_episodes == 4 and rep.objects[0].object == "cube"
    assert rep.objects[0].seen_rate == 1.0 and rep.objects[0].unseen_pos_p50 is None
    assert rnd.lo <= rep.obs_noise_std.lo <= rep.obs_noise_std.hi <= rnd.hi * 1.15
    assert "equivalent obs_noise_std" in rep.summary()


def test_error_model_needs_true_poses(stub_task):
    with pytest.raises(ValueError, match="true poses"):
        measure(make_stub_oracle, PickPlace(), stub_task, range(1))


# ---------------------------------------------------------------- MuJoCo with rendering


@pytest.fixture(scope="module")
def cube_env():
    pytest.importorskip("mujoco")
    from skillforge.sim import make_env

    env = make_env(load("franka_pick_cube_to_bin"))
    if env.render() is None:
        env.close()
        pytest.skip("offscreen rendering unavailable")
    yield env
    env.close()


def test_sim_frames_agree_with_mujoco_geometry(cube_env):
    cube_env.reset(3)
    src = SimFrameSource(cube_env, virtual=(overview_camera(),))
    true = cube_env.data.xpos[cube_env.model.body("cube").id].copy()
    try:
        for cam in ("overview", "front", "wrist"):
            f = src.capture(cam)
            assert f.rgb.shape == (480, 640, 3) and f.depth.shape == (480, 640)
            mask = f.masks["cube"]
            assert mask.sum() > 100, cam
            v, u = np.nonzero(mask)
            uv, z = project(f, true[None])
            assert np.linalg.norm(uv[0] - [u.mean() + 0.5, v.mean() + 0.5]) < 2.5, cam  # pixels
            pts = backproject(f, mask)
            assert np.linalg.norm(pts - true, axis=1).max() < 0.02 * np.sqrt(3) + 0.004, cam  # on the cube
            assert abs(np.percentile(pts[:, 2], 97) - 0.04) < 0.003, cam  # its top face is at z = 4 cm
    finally:
        src.close()


def test_depth_noise_is_seeded(cube_env):
    cube_env.reset(0)
    src = SimFrameSource(cube_env, virtual=(overview_camera(),), depth_noise=0.002)
    try:
        src.reseed(5)
        a = src.capture("overview").depth
        src.reseed(5)
        b = src.capture("overview").depth
        src.reseed(6)
        c = src.capture("overview").depth
        assert np.array_equal(a, b) and not np.array_equal(a, c)
    finally:
        src.close()


@pytest.mark.parametrize(
    "task_name, tol",
    [("franka_pick_cube_to_bin_hard", 0.003), ("franka_pick_cylinder_to_bin", 0.002), ("franka_pick_ball_to_bin", 0.003)],
)
def test_depth_estimator_finds_each_shape_at_rest(cube_env, task_name, tol):
    task = load(task_name)
    env = make_depth_env(task)
    try:
        for seed in (0, 1, 2):
            obs = env.reset(seed)
            for name, truth in env.true_poses().items():
                assert env.estimates[name].found
                assert np.linalg.norm(obs.objects[name].pos - truth.pos) < tol, (seed, name)
                if task.scene.object(name).shape == "box":
                    dyaw = wrap(quat_yaw(obs.objects[name].quat) - quat_yaw(truth.quat), np.pi / 2)
                    assert abs(dyaw) < np.radians(3), (seed, name)
    finally:
        env.close()


def test_depth_env_runs_an_episode_and_is_deterministic(cube_env):
    task = load("franka_pick_cube_to_bin")

    def run():
        env = make_depth_env(task)
        try:
            skill = PickPlace()
            obs = env.reset(11)
            skill.reset(task)
            trace, info = [obs.objects["cube"].pos.copy()], None
            for _ in range(task.max_steps):
                obs, info = env.step(skill.act(obs))
                trace.append(obs.objects["cube"].pos.copy())
                if info.success or info.terminated:
                    break
            return np.array(trace), info.success
        finally:
            env.close()

    (a, ok_a), (b, ok_b) = run(), run()
    assert ok_a and ok_b
    assert np.array_equal(a, b)


def test_color_segmentation_matches_the_sim_mask(cube_env):
    task = load("franka_pick_cube_to_bin")
    from skillforge.sim import make_env

    red = (0.85, 0.15, 0.15)
    env = PerceivedEnv(make_env(task), DepthEstimator(segment="color", colors={"cube": red}))
    try:
        obs = env.reset(2)
        truth = env.true_poses()["cube"]
        assert env.estimates["cube"].found
        assert np.linalg.norm(obs.objects["cube"].pos - truth.pos) < 0.004
    finally:
        env.close()


def test_yaw_quat_roundtrip():
    for y in (-1.0, 0.0, 0.4):
        assert abs(quat_yaw(yaw_quat(y)) - y) < 1e-12
