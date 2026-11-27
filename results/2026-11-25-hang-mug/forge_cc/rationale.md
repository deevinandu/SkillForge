I changed one number in `skill.py`: `grasp_depth` went from 0.018 to 0.028 m (plus the version bump to 0.1.1).
It targets `grasp_slip`, the most common failure, which in the report clustered at low friction and often high mass; my inference is that a deeper rim pinch better resists the mug tipping in the hand, though I did not measure contacts.
On 100 DEV seeds, success went from 88% (6 grasp_slip, 4 wrong_placement, 2 collision) to 100% (no failures, 95% CI 96%-100%).
This is measured in sim on DEV seeds only; the gate seeds and hardware are untested, and I tried no other depths, so the margin around 0.028 is unknown.
