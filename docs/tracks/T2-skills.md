# T2 · Skills (Phase 1)

**Goal:** a primitive library and grasp planner that make good seed skills, so agents (T3) start from something solid.

## Deliverables
1. `skills/primitives.py`: composable closed-loop primitives over `Action(mode="ee")`: `move_to`, `approach(above, hover)`, `descend_until(contact|height)`, `grip(settle)`, `lift`, `transport`, `place`, `release`, `retreat`. Each returns an action and a status (running/done/failed with a reason). A small phase-machine helper (`Sequence`, `Retry(n)`, `Fallback`) builds skills from them.
2. `skills/grasp.py`: grasp planning from `ObjectSpec` geometry, no learning yet. Box: candidate top-down grasps across each face pair that fits `max_gripper_width`, scored by width margin and alignment. Cylinder: across the diameter at several yaws. Sphere: top-down. Mesh: antipodal sampling on the trimesh surface (friction-cone check) as a stretch goal. Output: ranked `Grasp(pos, quat, width, score)` in object frame, plus a world-frame transform helper.
3. `skills/pick_place.py` v0.2: uses grasp.py for the approach orientation (yaw from the observed object quat), retries a missed grasp once, detects a slip (gripper width collapses to ~0) and re-grasps.
4. Keep every skill loadable as a `SkillSource` (single file, imports only `skillforge.*`, numpy). T3 needs that.
5. Tests: grasp planner on each shape (width limits respected, scores sorted); primitives on `StubEnv`; a slip injected in a stub subclass triggers re-grasp.

## Notes
- Until T1 lands, develop against `StubEnv`. When T1 merges, run on the Franka task and record the baseline success rate in HANDOVER.md.
- Don't change contracts; request changes in HANDOVER.md.
