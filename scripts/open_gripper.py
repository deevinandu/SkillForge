#!/usr/bin/env python3
"""Open the Franka gripper directly through franka_gripper's move action (not frankapy).

Safe to run while the frankapy bridge server is up: frankapy's node has a fixed name, so starting
another frankapy program would shut down the bridge's ROS link; this uses its own anonymous node.
Whatever the gripper holds is released and falls: hold it first.

    source ~/Documents/frankapy/catkin_ws/devel/setup.bash
    /usr/bin/python3 scripts/open_gripper.py [--width 0.08] [--speed 0.04]
"""

import argparse

import actionlib
import rospy
from franka_gripper.msg import MoveAction, MoveGoal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=0.08, help="opening in metres (0.08 = fully open)")
    ap.add_argument("--speed", type=float, default=0.04, help="m/s")
    ap.add_argument("--action", default="/franka_gripper_1/move")
    a = ap.parse_args()
    rospy.init_node("open_gripper", anonymous=True)
    client = actionlib.SimpleActionClient(a.action, MoveAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise SystemExit(f"no gripper action server at {a.action}: is the control stack running?")
    client.send_goal(MoveGoal(width=a.width, speed=a.speed))
    client.wait_for_result(rospy.Duration(10.0))
    print("gripper:", client.get_result())


if __name__ == "__main__":
    main()
