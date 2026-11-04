# Later tracks (outline)

- **T5 Learning (Phase 2):** episodes → LeRobot-format dataset (obs, actions, images); train diffusion policy and ACT on GPU (local RTX 5070 Ti, 12 GB); evaluate the learned policy as a `Skill` through the same harness; compare scripted vs learned under 2× randomization. Later: PPO fine-tuning in MJX, VLA fine-tuning (openpi / OpenVLA).
- **T6 Perception (Phase 3):** replace ground-truth object poses with estimated ones (FoundationPose / open-vocabulary detection + depth); VLM success checks from camera frames; the `obs_noise` axis becomes real perception error.
- **T7 Real Robot (Phase 3):** ROS 2 bridge implementing `SimEnv` on hardware (xArm 6 first, then Franka); system ID from logged trajectories; real-to-sim calibration updates `RandomizationSpec` centres and ranges; a sim2real scoreboard per skill.
- **T8 Scale and Research (Phase 4):** LLM task generation (scenes + success specs), a skill library across ~50 tasks, a public benchmark, a write-up (CoRL / ICRA workshop), open-source launch, DARWIN adapter.
