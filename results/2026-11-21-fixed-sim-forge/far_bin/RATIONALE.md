I found no change that beat the baseline, so skill.py behaves the same as before and only the version is now "0.1.1".
The common failures (collision with the table, wrong_placement, timeout) all appear after a grasp/release cycle, mostly at low friction_scale (0.33-0.42), so I suspected slip from fast acceleration during lift and transport.
I tried capping how far the EE target advances per step in transport (0.02 m/step gave 20/30, 0.05 m/step gave 25/30) and in lift alone (24/30). All were no better than the unmodified skill, whose dev success rate is 26/30 = 87% (the single-run results are noisy, ±15% CI).
I reverted these changes. The unused `slow` helper and `carry_step` attribute remain in skill.py but are not wired in.
Dev success rate before: 87% (26/30); after: 87% (26/30). The low-friction slip and re-grasp finger-table collisions remain unfixed.
