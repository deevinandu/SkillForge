# T6 · Perceiving dominoes

**Goal:** `ToppleChain` runs on estimated poses, and tells standing, misplaced and fallen
dominoes apart from cameras.

## Deliverables
1. `DepthEstimator` for thin boxes: yaw modulo pi (a domino is symmetric), position of an
   upright domino from its visible faces, **tilt** (fallen or not, and which way), which a
   top-down view alone gets wrong for a domino leaning toward the camera.
2. Error model on the domino tasks: position, yaw, tilt error; how often a domino moved by a
   disturbance is noticed within N steps.
3. `PerceivedEnv` + `ToppleChain` on the three tasks: success cost vs true poses, paired seeds.
4. Real cameras (with T7): the same estimator on the lab's cameras, calibrated.
