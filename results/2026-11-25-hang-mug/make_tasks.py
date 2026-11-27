"""Writes the mug-hanging task files from parameters (kept under runs/, the JSON is what's committed)."""
import json, math
from skillforge.contracts import TaskSpec

ROBOT = {"name": "franka_panda", "source": "menagerie:franka_emika_panda", "ee_site": "tcp",
         "gripper": "parallel", "max_gripper_width": 0.08}
MUG = {"name": "mug", "shape": "mug", "size": [0.0325, 0.045, 0.003], "mass": 0.2, "friction": 0.8,
       "handle": {"top": 0.025, "bottom": -0.02, "reach": 0.03, "radius": 0.004},
       "pose": {"pos": [0.42, -0.12, 0.045]}, "color": "white"}

def stand(cx, cy, pegs, post_h=0.34, peg_l=0.07, peg_r=0.005, tilt=math.radians(30), name="stand"):
    """A mug tree: square base, round post, pegs leaving the post at (height, azimuth) angled up by `tilt`."""
    objs = [
        {"name": f"{name}_base", "shape": "box", "size": [0.06, 0.06, 0.01], "mass": 1.0, "fixed": True,
         "pose": {"pos": [cx, cy, 0.01]}, "color": "orange"},
        {"name": f"{name}_post", "shape": "cylinder", "size": [0.012, post_h / 2, 0.0], "mass": 1.0, "fixed": True,
         "pose": {"pos": [cx, cy, post_h / 2]}, "color": "orange"},
    ]
    for pname, z, az in pegs:
        d = [math.cos(az) * math.cos(tilt), math.sin(az) * math.cos(tilt), math.sin(tilt)]
        root = [cx + 0.012 * math.cos(az), cy + 0.012 * math.sin(az), z]
        c = [root[i] + d[i] * peg_l / 2 for i in range(3)]
        # quaternion turning z onto d: axis z x d, angle acos(d_z)
        ax = [-d[1], d[0], 0.0]; n = math.hypot(*ax); ang = math.acos(d[2])
        q = [math.cos(ang / 2)] + [a / n * math.sin(ang / 2) for a in ax]
        objs.append({"name": pname, "shape": "cylinder", "size": [peg_r, peg_l / 2, 0.0], "mass": 0.05, "fixed": True,
                     "friction": 0.6, "pose": {"pos": [round(v, 6) for v in c], "quat": [round(v, 6) for v in q]},
                     "color": "orange"})
    return objs

def task(tid, instruction, objects, support, rnd, max_steps=1000):
    t = {"id": tid, "instruction": instruction,
         "scene": {"robot": ROBOT, "objects": objects, "randomization": rnd},
         "success": {"kind": "object_hung", "object": "mug", "support": support, "hold_s": 1.0, "max_speed": 0.15},
         "max_steps": max_steps, "control_hz": 20.0}
    TaskSpec.model_validate(t)
    return t

RND = {"object_xy_jitter": {"lo": 0.0, "hi": 0.03}, "object_yaw_jitter": {"lo": math.pi, "hi": math.pi},
       "friction_scale": {"lo": 0.8, "hi": 1.2}, "mass_scale": {"lo": 0.75, "hi": 1.25},
       "obs_noise_std": {"lo": 0.0, "hi": 0.002}}
RND_FIXED_HANDLE = {**RND, "object_yaw_jitter": {"lo": 0.0, "hi": 0.0}}

# v1: one peg, pointing back at the robot (-x)
one = stand(0.60, 0.12, [("peg", 0.24, math.pi)])
# v3: a tree, three pegs at different heights and azimuths; the target is the middle one
tree = stand(0.60, 0.10, [("peg_left", 0.22, math.pi - 1.2), ("peg_front", 0.25, math.pi), ("peg_right", 0.28, math.pi + 1.2)])

out = {
  "franka_hang_mug": task("franka_hang_mug",
      "Pick up the mug and hang it by its handle on the peg of the mug stand.", [MUG, *one], "peg", RND),
  "franka_hang_mug_handle_fixed": task("franka_hang_mug_handle_fixed",
      "Pick up the mug and hang it by its handle on the peg of the mug stand.",
      [{**MUG, "pose": {"pos": [0.42, -0.12, 0.045], "quat": [0.7071068, 0, 0, -0.7071068]}}, *one], "peg", RND_FIXED_HANDLE),
  "franka_hang_mug_tree": task("franka_hang_mug_tree",
      "Pick up the mug and hang it by its handle on the front peg of the mug tree.", [MUG, *tree], "peg_front", RND),
}
if __name__ == "__main__":
    for k, t in out.items():
        json.dump(t, open(f"tasks/{k}.json", "w"), indent=2); print("wrote", k)
