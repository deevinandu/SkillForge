#!/usr/bin/env bash
# SkillForge launcher for the lab can pyramid (franka_stack_can_pyramid_lab).
#
#   ./run.sh test            unit tests + lint (no robot)
#   ./run.sh dry [SEEDS]     full session on the simulated twin (no robot), e.g. ./run.sh dry 0-4
#   ./run.sh stack           start the control PC (franka_interface + ROS nodes, opens terminals)
#   ./run.sh real [SEEDS]    REAL ARM: starts the frankapy bridge, runs the trial, stops the bridge
#   ./run.sh camera          grab one RealSense frame with the calibration overlay
#
# Real-arm order: robot unlocked and BLUE -> ./run.sh stack -> ./run.sh real.
# A person must be at the e-stop. Ctrl-C this script BEFORE pressing the e-stop or grey buttons
# (doing it mid-skill crashes franka_interface; then restart from ./run.sh stack).
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
REPO=$PWD
TASK=franka_stack_can_pyramid_lab
SKILL=0.2.0
EXTRINSICS=runs/real/handeye/2026-11-27/T_world_cam.json
FRANKAPY=$HOME/Documents/frankapy
CONTROL_PC=iam-sleepy
PORT=8765
PY=$REPO/.venv/bin/python

need_venv() {
    if [[ ! -x $PY ]]; then
        echo "no .venv: run  python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,sim]'" >&2
        exit 1
    fi
}

cmd=${1:-help}
shift || true

case $cmd in
test)
    need_venv
    .venv/bin/ruff check src tests
    .venv/bin/pytest -q
    ;;

dry)
    need_venv
    "$PY" -m skillforge.real run "$TASK" --skill "$SKILL" --seeds "${1:-0-0}" --dry-run
    ;;

stack)
    bash "$FRANKAPY/bash_scripts/start_control_pc.sh" -u student -i "$CONTROL_PC"
    echo "Wait until franka_interface says it is ready, then: ./run.sh real"
    ;;

real)
    need_venv
    if pgrep -f frankapy_bridge_server.py >/dev/null; then
        echo "A bridge is already running (only one frankapy node may exist). Stop it first:" >&2
        pgrep -af frankapy_bridge_server.py >&2
        exit 1
    fi
    read -r -p "REAL ARM. Robot blue, workspace clear, someone at the e-stop? [y/N] " ok
    [[ $ok == [yY] ]] || { echo "aborted"; exit 1; }

    set +u  # ROS setup scripts reference unset variables
    # shellcheck disable=SC1091
    source "$FRANKAPY/catkin_ws/devel/setup.bash"
    set -u
    stamp=$(date +%H%M%S)
    log=runs/real/bridge_server_$stamp.out
    PYTHONPATH=$FRANKAPY:${PYTHONPATH:-} /usr/bin/python3 scripts/frankapy_bridge_server.py \
        --port "$PORT" --log runs/real/bridge.jsonl >"$log" 2>&1 &
    bridge=$!
    trap 'echo "stopping bridge ($bridge)"; kill -INT $bridge 2>/dev/null; wait $bridge 2>/dev/null || true' EXIT

    echo "starting bridge (log: $log) ..."
    for _ in $(seq 60); do
        grep -q "bridge listening" "$log" && break
        kill -0 "$bridge" 2>/dev/null || { echo "bridge died:" >&2; tail -20 "$log" >&2; exit 1; }
        sleep 1
    done
    grep -q "bridge listening" "$log" || { echo "bridge not up after 60 s:" >&2; tail -20 "$log" >&2; exit 1; }

    "$PY" -m skillforge.real run "$TASK" --skill "$SKILL" --seeds "${1:-0-0}" \
        --extrinsics "$EXTRINSICS" --port "$PORT" --free-placement
    ;;

camera)
    need_venv
    "$PY" -m skillforge.real camera --extrinsics "$EXTRINSICS"
    ;;

*)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
