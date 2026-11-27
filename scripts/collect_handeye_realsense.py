#!/usr/bin/env python3
"""Hand-eye data for a fixed RealSense camera and a Franka with an ArUco marker on the gripper.

Standalone on purpose: frankapy runs on the lab's Python 3.8 and SkillForge needs 3.10, so this
script does not import skillforge. Run it with the Python that has frankapy, after
`start_control_pc.sh`, with a person at the e-stop:

    /usr/bin/python3 scripts/collect_handeye_realsense.py --out runs/real/handeye/<date>

The arm goes into guide mode. Move it by hand so the gripper tag (default DICT_4X4_50 id 5, about 2.45 cm) is in view,
press 's' in the window to save an image + end-effector pose pair, and 'q' to stop. Take 15-20
poses with large rotations, not just translations, spread over the part of the table the task uses.

Output (the layout team 7's compute_calibration.py / compute_spatial_calibration.py read):
    <out>/calib_data/K.npy, D.npy
    <out>/calib_data/images/NNN.png, NNN_T_base_ee.npy, NNN_depth.npy (uint16, x meta depth_scale = m)
"""

import argparse
import json
import os

import numpy as np
import pyrealsense2 as rs


def _gui_cv2():
    """An OpenCV that can open windows.

    On the lab workstation a pip `opencv-python-headless` in ~/.local shadows the GTK build that
    ROS Noetic installs in /usr/lib/python3/dist-packages. Two OpenCVs must never both be imported
    (their submodules mix), so the system build is loaded first when it exists and the pip one
    is not touched."""
    import glob
    import importlib.machinery
    import importlib.util
    import sys

    for path in glob.glob("/usr/lib/python3/dist-packages/cv2*.so"):
        loader = importlib.machinery.ExtensionFileLoader("cv2", path)
        spec = importlib.util.spec_from_file_location("cv2", path, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules["cv2"] = mod
        print(f"using system OpenCV {mod.__version__} ({path})")
        return mod
    import cv2

    try:
        cv2.namedWindow("_probe")
        cv2.destroyWindow("_probe")
    except cv2.error:  # headless build: "The function is not implemented"
        raise SystemExit("this OpenCV cannot open windows: pip uninstall opencv-python-headless") from None
    return cv2


cv2 = _gui_cv2()

def make_detector(dict_name: str):
    """detect(img) -> (corners, ids, rejected) for one predefined ArUco dictionary."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return det.detectMarkers
    params = cv2.aruco.DetectorParameters_create()
    return lambda img: cv2.aruco.detectMarkers(img, d, parameters=params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--serial", help="camera serial; default: the only one connected")
    ap.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary of the gripper tag")
    ap.add_argument("--marker-id", type=int, default=5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    data = os.path.join(a.out, "calib_data")
    os.makedirs(os.path.join(data, "images"), exist_ok=True)

    cfg = rs.config()
    if a.serial:
        cfg.enable_device(a.serial)
    detect = make_detector(a.dict)
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, 30)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = float(prof.get_device().first_depth_sensor().get_depth_scale())
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])
    D = np.array(intr.coeffs, dtype=float)
    np.save(os.path.join(data, "K.npy"), K)
    np.save(os.path.join(data, "D.npy"), D)
    serial = prof.get_device().get_info(rs.camera_info.serial_number)
    print(f"camera {serial}\nK =\n{K}\nD = {D}\nlooking for {a.dict} id {a.marker_id}")
    with open(os.path.join(data, "meta.json"), "w") as fh:
        json.dump({"serial": serial, "dict": a.dict, "marker_id": a.marker_id, "size": [a.width, a.height],
                   "depth_scale": depth_scale}, fh, indent=2)

    # imported after the camera starts, so a camera problem fails before the arm moves
    from frankapy import FrankaArm

    fa = FrankaArm()
    fa.run_guide_mode(duration=1e6, block=False)
    print("\nGuide mode ON: move the arm by hand. In the window: 's' save, 'q' quit.\n")
    i = len([f for f in os.listdir(os.path.join(data, "images")) if f.endswith(".png")])
    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth_frame:
                continue
            img = np.asanyarray(color.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()  # uint16, x depth_scale = metres, aligned to img
            corners, ids, _ = detect(img)
            seen = ids is not None and a.marker_id in ids.flatten()
            disp = img.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(disp, corners, ids)
            cv2.putText(disp, f"saved {i}  marker {'SEEN' if seen else 'not seen'}  [s]ave [q]uit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if seen else (0, 0, 255), 2)
            cv2.imshow("handeye", disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                if not seen:
                    print("marker not in view: not saved")
                    continue
                T = fa.get_pose().matrix
                cv2.imwrite(os.path.join(data, "images", f"{i:03d}.png"), img)
                np.save(os.path.join(data, "images", f"{i:03d}_T_base_ee.npy"), T)
                np.save(os.path.join(data, "images", f"{i:03d}_depth.npy"), depth)
                print(f"[{i}] saved, ee at {T[:3, 3].round(3)}")
                i += 1
    finally:
        fa.stop_skill()
        pipe.stop()
        cv2.destroyAllWindows()
    print(f"\n{i} pairs in {data}" + ("  (fewer than 15: the fit will be poor)" if i < 15 else ""))


if __name__ == "__main__":
    main()
