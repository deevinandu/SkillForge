"""T1: the MuJoCo Franka env. Skips cleanly when mujoco or the menagerie assets are unavailable."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from skillforge.sim import default_mujoco_gl  # noqa: E402

default_mujoco_gl()  # before mujoco's first import, so headless rendering picks osmesa
mujoco = pytest.importorskip("mujoco")

from skillforge.contracts import Action, RandomizationSpec, Range, SimEnv, TaskSpec  # noqa: E402
from skillforge.eval import evaluate  # noqa: E402
from skillforge.sim import is_success, make_env  # noqa: E402
from skillforge.sim.assets import AssetError, resolve_robot_source  # noqa: E402
from skillforge.skills import PickPlace  # noqa: E402
from skillforge.skills.pick_place_v01 import PickPlaceV01  # noqa: E402

TASKS = Path(__file__).parent.parent / "tasks"

try:
    resolve_robot_source("menagerie:franka_emika_panda")
except (AssetError, OSError) as e:  # offline, no git, ...
    pytest.skip(f"menagerie assets unavailable: {e}", allow_module_level=True)

from skillforge.sim.mujoco_env import MujocoEnv  # noqa: E402

DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def load(name: str) -> TaskSpec:
    return TaskSpec.model_validate(json.loads((TASKS / f"{name}.json").read_text()))


def with_randomization(task: TaskSpec, **fields) -> TaskSpec:
    scene = task.scene.model_copy(update={"randomization": RandomizationSpec(**fields)})
    return task.model_copy(update={"scene": scene})


@pytest.fixture(scope="module")
def task() -> TaskSpec:
    return load("franka_pick_cube_to_bin")


@pytest.fixture(scope="module")
def plain_task(task) -> TaskSpec:
    return with_randomization(task)


@pytest.fixture(scope="module")
def env(plain_task):
    e = MujocoEnv(plain_task)
    yield e
    e.close()


def hold(env, obs, n, **kw):
    a = Action(ee_pos=obs.ee_pos.copy(), ee_quat=obs.ee_quat.copy(), **kw)
    events = []
    for _ in range(n):
        obs, info = env.step(a)
        events += info.events
    return obs, info, events


# ------------------------------------------------------------------ basics


def test_model_loads_and_satisfies_protocol(env, plain_task):
    assert isinstance(env, SimEnv)
    assert isinstance(make_env(plain_task), MujocoEnv)
    obs = env.reset(0)
    assert obs.qpos.shape == (7,) and obs.qvel.shape == (7,)
    assert obs.ee_pos.shape == (3,) and obs.ee_quat.shape == (4,)
    assert np.allclose(np.abs(obs.ee_quat), DOWN, atol=1e-3)  # tcp starts pointing down
    assert obs.gripper_width == pytest.approx(0.08, abs=1e-6)
    assert set(obs.objects) == {"cube"}
    assert obs.contact_force is not None and obs.contact_force.shape == (3,)
    assert env.n_substeps == 15  # 20 Hz control on a 3.33 ms physics step
    assert env.n_substeps * env.model.opt.timestep == pytest.approx(env.dt)  # exact control period


def test_home_pose_holds_still(env):
    obs0 = env.reset(0)
    obs, _, events = hold(env, obs0, 40)
    assert np.linalg.norm(obs.ee_pos - obs0.ee_pos) < 2e-3
    assert np.max(np.abs(obs.qpos - obs0.qpos)) < 1e-2
    assert np.max(np.abs(obs.qvel)) < 0.05
    assert not [e for e in events if e.startswith("collision")]


@pytest.mark.parametrize("offset", [(0.0, 0.0, -0.3), (-0.1, 0.2, -0.25), (0.05, -0.25, -0.4)])
def test_ee_reaches_target_within_5mm(env, offset):
    obs = env.reset(0)
    target = obs.ee_pos + np.array(offset)
    for _ in range(60):
        obs, info = env.step(Action(ee_pos=target, ee_quat=DOWN))
    assert np.linalg.norm(obs.ee_pos - target) < 5e-3, obs.ee_pos - target
    assert "ik_failure" not in info.events


def test_ee_orientation_tracks_yaw(env):
    obs = env.reset(0)
    yaw = 0.5
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    target_q = np.zeros(4)
    mujoco.mju_mulQuat(target_q, qz, DOWN)
    for _ in range(40):
        obs, _ = env.step(Action(ee_pos=obs.ee_pos, ee_quat=target_q))
    assert abs(np.dot(obs.ee_quat, target_q)) > 0.9999


def test_joint_mode_passes_targets_through(env):
    obs = env.reset(0)
    q = obs.qpos.copy()
    q[0] += 0.3
    for _ in range(40):
        obs, _ = env.step(Action(mode="joint", joint_targets=q))
    assert np.max(np.abs(obs.qpos - q)) < 5e-3


def test_gripper_maps_zero_to_open_one_to_closed(env):
    obs = env.reset(0)
    obs, _, events = hold(env, obs, 20, gripper=1.0)
    assert obs.gripper_width < 0.004
    assert "grasp_miss" in events  # closed on nothing
    obs, _, _ = hold(env, obs, 20, gripper=0.0)
    assert obs.gripper_width > 0.075


# ------------------------------------------------------------------ determinism and randomization

FULL = dict(
    object_xy_jitter=Range(lo=0.02, hi=0.05),
    object_yaw_jitter=Range(lo=0.2, hi=0.6),
    friction_scale=Range(lo=0.5, hi=1.5),
    mass_scale=Range(lo=0.5, hi=2.0),
    actuator_gain_scale=Range(lo=0.7, hi=1.3),
    obs_noise_std=Range(lo=0.002, hi=0.005),
    action_latency_steps=Range(lo=0, hi=3),
)


def rollout(env, task, seed, n=80):
    sk = PickPlace()
    obs = env.reset(seed)
    sk.reset(task)
    trace, events = [], []
    for _ in range(n):
        obs, info = env.step(sk.act(obs))
        trace.append(np.concatenate([obs.qpos, obs.ee_pos, obs.objects["cube"].pos]))
        events += info.events
    return np.array(trace), events, env.randomization_sample()


def test_same_seed_same_episode(task):
    t = with_randomization(task, **FULL)
    env = MujocoEnv(t)
    a = rollout(env, t, 7)
    b = rollout(env, t, 7)
    rollout(env, t, 8)  # a different episode in between must not leak into the next
    c = rollout(env, t, 7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert a[1] == b[1] == c[1] and a[2] == b[2] == c[2]
    d = rollout(env, t, 8)
    assert not np.array_equal(a[0], d[0])
    # a fresh env instance gives the same episode too
    assert np.array_equal(rollout(MujocoEnv(t), t, 7)[0], a[0])


def test_each_randomization_field_changes_the_model(task):
    nominal = MujocoEnv(with_randomization(task))
    nominal.reset(0)
    nm = nominal.model
    t = with_randomization(task, **FULL)
    env = MujocoEnv(t)
    m, d = env.model, env.data
    cube_b, cube_g = m.body("cube").id, m.geom("cube").id
    seen = []
    for seed in (0, 1):
        obs = env.reset(seed)
        s = env.randomization_sample()
        seen.append(s)
        # xy and yaw jitter move the object's true pose
        start = np.asarray(task.scene.object("cube").pose.pos)
        assert np.allclose(d.xpos[cube_b][:2] - start[:2], [s["cube.dx"], s["cube.dy"]], atol=1e-9)
        assert max(abs(s["cube.dx"]), abs(s["cube.dy"])) <= s["object_xy_jitter"]
        yaw = 2 * np.arctan2(d.xquat[cube_b][3], d.xquat[cube_b][0])
        assert yaw == pytest.approx(s["cube.dyaw"], abs=1e-9) and abs(yaw) <= s["object_yaw_jitter"]
        # friction and mass on the object; robot untouched
        assert m.geom_friction[cube_g, 0] == pytest.approx(nm.geom_friction[cube_g, 0] * s["friction_scale"])
        assert m.body_mass[cube_b] == pytest.approx(nm.body_mass[cube_b] * s["mass_scale"])
        hand = m.body("hand").id
        assert m.body_mass[hand] == nm.body_mass[hand]
        # actuator gains (kp and kv) on every actuator
        k = s["actuator_gain_scale"]
        assert np.allclose(m.actuator_gainprm[:, 0], nm.actuator_gainprm[:, 0] * k)
        assert np.allclose(m.actuator_biasprm[:, 1:3], nm.actuator_biasprm[:, 1:3] * k)
        # observation noise: observed != true, true is what success uses
        assert 0 < np.linalg.norm(obs.objects["cube"].pos - d.xpos[cube_b]) < 0.05
        assert s["action_latency_steps"] in (0.0, 1.0, 2.0, 3.0)
    for key in ("object_xy_jitter", "object_yaw_jitter", "friction_scale", "mass_scale", "actuator_gain_scale", "obs_noise_std"):
        assert seen[0][key] != seen[1][key], key


@pytest.mark.parametrize("lag", [0, 3])
def test_action_latency_delays_commands(task, lag):
    t = with_randomization(task, action_latency_steps=Range.fixed(lag))
    env = MujocoEnv(t)
    obs = env.reset(0)
    q0 = env.data.ctrl[:7].copy()
    target = obs.ee_pos + [0.0, 0.0, -0.2]
    changed_at = None
    for i in range(6):
        env.step(Action(ee_pos=target, ee_quat=DOWN))
        if changed_at is None and not np.allclose(env.data.ctrl[:7], q0):
            changed_at = i
    assert changed_at == lag


# ------------------------------------------------------------------ success, events, failure modes


def teleport(env, name, pos):
    a = env.model.joint(f"{name}_free").qposadr[0]
    env.data.qpos[a : a + 3] = pos
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def test_success_judged_on_true_state(task):
    t = with_randomization(task, obs_noise_std=Range.fixed(0.2))  # observations are useless
    env = MujocoEnv(t)
    obs = env.reset(0)
    _, info, _ = hold(env, obs, 1)
    assert not info.success
    bin_ = t.scene.region("bin")
    teleport(env, "cube", [bin_.center[0], bin_.center[1], 0.02])
    obs, info, _ = hold(env, obs, 3)
    true = {"cube": env.data.xpos[env.model.body("cube").id].copy()}
    assert info.success and is_success(t, true, env._start)
    assert not is_success(t, {"cube": obs.objects["cube"].pos}, env._start)  # the noisy view disagrees


def test_lift_task_success(task):
    lift = load("franka_lift_cube")
    env = MujocoEnv(with_randomization(lift))
    obs = env.reset(0)
    _, info, _ = hold(env, obs, 1)
    assert not info.success
    teleport(env, "cube", [0.5, 0.0, 0.02 + lift.success.min_height + 0.03])  # free fall: ~1 cm per step
    obs, info = env.step(Action(ee_pos=obs.ee_pos, ee_quat=DOWN))
    assert info.success


def test_object_off_table_terminates(env):
    obs = env.reset(0)
    teleport(env, "cube", [0.5, 1.5, -0.2])
    _, info, _ = hold(env, obs, 1)
    assert info.terminated and not info.success


def test_grasp_event_and_scripted_pick(env, plain_task):
    sk = PickPlace()
    obs = env.reset(0)
    sk.reset(plain_task)
    events = []
    for _ in range(plain_task.max_steps):
        obs, info = env.step(sk.act(obs))
        events += info.events
        if info.success:
            break
    assert "grasp:cube" in events
    assert info.success, events
    assert not [e for e in events if e.startswith("collision") or e == "ik_failure"]


def test_grip_force_is_realistic(env, plain_task):
    # ~10 N per finger on the 4 cm cube (menagerie default was ~1 N; the real hand does 70 N)
    sk = PickPlace()
    obs = env.reset(0)
    sk.reset(plain_task)
    for _ in range(plain_task.max_steps):
        obs, info = env.step(sk.act(obs))
        if any(e.startswith("grasp:") for e in info.events):
            break
    _, _, _ = hold(env, obs, 4, gripper=1.0)
    m, d = env.model, env.data
    f = np.zeros(6)
    normal = {b: 0.0 for b in env._finger_bodies}
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact.geom[i]]
        fb = b1 if b1 in normal else b2 if b2 in normal else None
        if fb is not None and m.body("cube").id in (b1, b2):
            mujoco.mj_contactForce(m, d, i, f)
            normal[fb] += f[0]
    print("finger normal forces (N):", normal)
    assert all(5.0 < n < 30.0 for n in normal.values()), normal


def test_collision_event_when_pushing_into_table(env):
    obs = env.reset(0)
    target = np.array([0.45, 0.25, -0.05])  # below the table top, away from the cube
    events = []
    for _ in range(80):
        obs, info = env.step(Action(ee_pos=target, ee_quat=DOWN))
        events += info.events
    assert any(e.startswith("collision:") and e.endswith("-table") for e in events), events


def test_ik_failure_event_on_unreachable_target(env):
    obs = env.reset(0)
    events = []
    for _ in range(60):
        obs, info = env.step(Action(ee_pos=np.array([2.0, 0.0, 0.5]), ee_quat=DOWN))
        events += info.events
    assert "ik_failure" in events
    assert events.count("ik_failure") < 10  # reported on onset, not every step
    assert np.all(np.isfinite(obs.qpos))


def test_grasp_slip_when_object_pulled_away(env, plain_task):
    sk = PickPlaceV01()  # uses the v0.1 phase attribute
    obs = env.reset(0)
    sk.reset(plain_task)
    events = []
    for _ in range(200):
        obs, info = env.step(sk.act(obs))
        events += info.events
        if sk.phase == "transport":
            break
    assert "grasp:cube" in events
    teleport(env, "cube", [0.2, -0.4, 0.02])  # yank it out of the hand
    _, _, ev = hold(env, obs, 2, gripper=1.0)
    assert "grasp_slip" in ev


def test_sim_instability_is_reported_and_terminates(env):
    obs = env.reset(0)
    env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1  # as if MuJoCo just reset itself
    obs, info, _ = hold(env, obs, 1)
    assert "sim_unstable" in info.events and info.terminated
    _, info, _ = hold(env, obs, 1)
    assert info.terminated and not info.success  # stays dead until reset
    env.reset(0)
    _, info, _ = hold(env, env.reset(0), 1)
    assert not info.terminated


# ------------------------------------------------------------------ eval, speed, rendering


def test_scripted_pickplace_beats_zero_on_randomized_franka_task(task):
    report, _ = evaluate(make_env, PickPlace(), task, range(10))
    print(report.summary())
    assert report.success_rate > 0.0, report.summary()


def test_speed_is_reported(env, plain_task):
    obs = env.reset(0)
    a = Action(ee_pos=obs.ee_pos + [0, 0, -0.2], ee_quat=DOWN)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    rate = n / (time.perf_counter() - t0)
    print(f"MujocoEnv: {rate:.0f} control steps/s (free motion, no rendering)")
    assert rate > 100  # a loose floor; the real number goes in HANDOVER.md


def test_render_returns_frame_or_none(env):
    env.reset(0)
    for cam in (None, "wrist", "front"):
        img = env.render(cam)
        assert img is None or (img.shape == (240, 320, 3) and img.dtype == np.uint8)


def test_render_degrades_gracefully_without_gl(plain_task, monkeypatch):
    def no_gl(*a, **k):
        raise RuntimeError("no OpenGL here")

    from skillforge.sim import mujoco_env

    monkeypatch.setattr(mujoco, "Renderer", no_gl)
    monkeypatch.setattr(mujoco_env, "_GL_BROKEN", None)  # restored after the test
    with pytest.warns(UserWarning, match="rendering unavailable"):
        env = MujocoEnv(plain_task, obs_cameras=("wrist",))
    obs = env.reset(0)
    assert env.render() is None and obs.images == {}
    obs, _ = env.step(Action(ee_pos=obs.ee_pos, ee_quat=DOWN))  # stepping still works
    assert obs.images == {}


def test_obs_cameras_fill_images_when_gl_works(plain_task):
    env = MujocoEnv(plain_task, obs_cameras=("wrist",), render_size=(48, 64))
    obs = env.reset(0)
    if env.render() is None:
        pytest.skip("no OpenGL backend on this machine")
    assert obs.images["wrist"].shape == (48, 64, 3)
    env.close()


def test_object_ejection_is_flagged(env):
    obs = env.reset(0)
    a = env.model.joint("cube_free").dofadr[0]
    env.data.qvel[a : a + 3] = [0.0, 0.0, 6.0]  # what a solver ejection looks like
    _, _, events = hold(env, obs, 1)
    assert "object_ejected:cube" in events


@pytest.mark.parametrize("scale", [0.7, 1.5])
def test_mass_randomization_updates_derived_constants(task, scale):
    # body_invweight0 scales constraint regularisation; it must follow the randomized mass
    nominal = MujocoEnv(with_randomization(task))
    env = MujocoEnv(with_randomization(task, mass_scale=Range.fixed(scale)))
    env.reset(3)
    b = env.model.body("cube").id
    assert env.model.body_invweight0[b] == pytest.approx(nominal.model.body_invweight0[b] / scale, rel=1e-6)


def test_tumbling_cube_lands_without_ejection(task):
    """Regression for the object_ejected artifact: random tumbling drops onto the table.

    Before mj_setConst ran after mass randomization, ~30% of these drops flung the cube
    faster than EJECT_SPEED (often to NaN). Now none may.
    """
    from skillforge.sim.mujoco_env import EJECT_SPEED

    env = MujocoEnv(with_randomization(task, mass_scale=Range(lo=0.7, hi=1.5)))
    m, d = env.model, env.data
    qa, va = env._obj_free["cube"], env._obj_dof["cube"]
    worst = 0.0
    for i in range(30):
        rng = np.random.default_rng(100 + i)
        env.reset(i)
        q = rng.normal(size=4)
        d.qpos[qa : qa + 3] = [0.5, -0.15, 0.05 + 0.08 * rng.random()]
        d.qpos[qa + 3 : qa + 7] = q / np.linalg.norm(q)
        d.qvel[va : va + 3] = rng.normal(0, 0.5, 3)
        d.qvel[va + 3 : va + 6] = rng.normal(0, 15, 3)
        for _ in range(500):
            mujoco.mj_step(m, d)
            if d.xpos[m.body("cube").id][2] < 0.06:
                worst = max(worst, float(np.linalg.norm(d.qvel[va : va + 3])))
    assert np.isfinite(worst) and worst < EJECT_SPEED, worst


def test_mesh_and_primitive_objects_build(plain_task, tmp_path):
    obj = tmp_path / "tri_prism.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\n"
        "f 1 3 2\nf 4 5 6\nf 1 2 5\nf 1 5 4\nf 2 3 6\nf 2 6 5\nf 3 1 4\nf 3 4 6\n"
    )
    d = json.loads(plain_task.model_dump_json())
    d["scene"]["objects"] += [
        {"name": "can", "shape": "cylinder", "size": [0.03, 0.05, 0], "pose": {"pos": [0.4, 0.1, 0.05]}},
        {"name": "ball", "shape": "sphere", "size": [0.025, 0, 0], "pose": {"pos": [0.6, 0.1, 0.025]}},
        {"name": "wedge", "shape": "mesh", "mesh": str(obj), "size": [0.04, 0.04, 0.04], "pose": {"pos": [0.3, -0.2, 0.0]}},
        {"name": "post", "shape": "box", "size": [0.01, 0.01, 0.05], "fixed": True, "pose": {"pos": [0.3, 0.3, 0.05]}},
    ]
    env = MujocoEnv(TaskSpec.model_validate(d))
    obs = env.reset(3)
    assert set(obs.objects) == {"cube", "can", "ball", "wedge", "post"}
    obs, _, _ = hold(env, obs, 20)
    rest = {"can": 0.05, "ball": 0.025, "post": 0.05}
    for name, z in rest.items():  # resting objects stay put on the table
        assert abs(obs.objects[name].pos[2] - z) < 2e-3, name
    assert obs.objects["wedge"].pos[2] > -0.01  # mesh collides with the table


def test_hard_task_is_wider_than_the_base_task(task):
    hard = load("franka_pick_cube_to_bin_hard")
    rb, rh = task.scene.randomization, hard.scene.randomization
    assert rh.object_xy_jitter.hi > rb.object_xy_jitter.hi
    assert rh.friction_scale.lo < rb.friction_scale.lo
    assert rh.obs_noise_std.hi > rb.obs_noise_std.hi
    assert rh.action_latency_steps.hi > rb.action_latency_steps.hi
    assert hard.scene.object("cube").mass > task.scene.object("cube").mass
    assert hard.scene.region("bin").half_extents[0] < task.scene.region("bin").half_extents[0]
    env = MujocoEnv(hard)
    obs = env.reset(5)
    _, info = env.step(Action(ee_pos=obs.ee_pos, ee_quat=DOWN))
    assert set(obs.objects) == {"cube"} and not info.terminated


def test_stacked_success_on_true_state(task):
    """object_on_object / all_of in the real env: a cube stacked on another, in and out of the bin."""
    d = json.loads(with_randomization(task).model_dump_json())
    cube = d["scene"]["objects"][0]
    top_z = cube["pose"]["pos"][2] + 2 * cube["size"][2]
    d["scene"]["objects"].append({**cube, "name": "top", "pose": {"pos": [*cube["pose"]["pos"][:2], top_z]}})
    on = {"kind": "object_on_object", "object": "top", "support": "cube"}
    d["success"] = {"kind": "all_of", "conditions": [on, {"kind": "object_in_region", "object": "cube", "region": "bin"}]}
    t = TaskSpec.model_validate(d)
    env = MujocoEnv(t)
    try:
        obs = env.reset(0)
        obs, info, _ = hold(env, obs, 3)
        assert not info.success  # stacked, but not in the bin
        c = t.scene.region("bin").center
        teleport(env, "cube", [c[0], c[1], 0.02])
        teleport(env, "top", [c[0], c[1], 0.06])
        obs, info, _ = hold(env, obs, 5)
        assert info.success
        teleport(env, "top", [c[0] + 0.08, c[1], 0.02])  # knocked off, next to it
        obs, info, _ = hold(env, obs, 5)
        assert not info.success
    finally:
        env.close()
