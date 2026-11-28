#!/usr/bin/env bash
# Move the arm to frankapy's home and open the gripper, then check it actually got there.
#   home joints: [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]   (the lab handout's home, used before power-down)
#
#   ./reset.sh            home + open gripper
#   ./reset.sh --close    home + close gripper
#
# Needs: control stack running (./run.sh stack), robot BLUE, someone at the e-stop.
# Whatever the gripper holds is released: hold it first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
FRANKAPY=$HOME/Documents/frankapy

# Only one frankapy program may run at a time (fixed node name /franka_arm_client).
if pgrep -f frankapy_bridge_server.py >/dev/null; then
    echo "The frankapy bridge is running; a second frankapy program would kill its ROS link." >&2
    echo "Stop ./run.sh real (Ctrl-C) first, then rerun ./reset.sh." >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source "$FRANKAPY/catkin_ws/devel/setup.bash"
set -u
export PYTHONPATH=$FRANKAPY:${PYTHONPATH:-}

args=()
[[ ${1:-} == --close ]] && args+=(-c)

echo "Homing the arm (frankapy reset_arm.py) ..."
/usr/bin/python3 "$FRANKAPY/scripts/reset_arm.py" "${args[@]}"

# frankapy can report a move finished before it is; wait for the joints themselves.
/usr/bin/python3 - <<'EOF'
import math, sys, time
import rospy
from franka_interface_msgs.msg import RobotState

HOME = [0, -math.pi/4, 0, -3*math.pi/4, 0, math.pi/2, math.pi/4]
rospy.init_node("reset_check", anonymous=True)
deadline = time.time() + 15
while True:
    q = rospy.wait_for_message("/robot_state_publisher_node_1/robot_state", RobotState, timeout=5).q
    err = max(abs(a - b) for a, b in zip(q, HOME))
    if err < 0.01 or time.time() > deadline:
        break
    time.sleep(0.2)
if err < 0.01:
    print(f"arm at frankapy home (max joint error {err:.4f} rad)")
else:
    print(f"arm NOT at home: max joint error {err:.3f} rad (wall, reflex or e-stop?)", file=sys.stderr)
    print("joints:", [round(x, 3) for x in q], file=sys.stderr)
    sys.exit(1)
EOF
