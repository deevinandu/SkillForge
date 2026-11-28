#!/usr/bin/env python3
"""Solve the fixed-camera (eye-to-hand) calibration from `collect_handeye_realsense.py` data.

    python3 scripts/solve_handeye.py runs/real/handeye/<date> --marker-size 0.025

Needs only numpy and an OpenCV with aruco (the system one on the lab box works). Reads the
dictionary and id from the collector's meta.json. For each saved pair it finds the tag
(solvePnP), then solves T_base_cam with OpenCV's hand-eye methods and keeps the one whose
residual is smallest. Writes <dir>/T_world_cam.json (world = robot base) for SkillForge's
`load_extrinsics`, and T_base_cam.npy for the team scripts.

Residual: with a correct T_base_cam, the tag's pose relative to the hand is the same in every
pair. The spread of that pose (translation, mm; rotation, deg) across pairs is the check. A few
mm is good; over ~10 mm means bad pairs (blur, a tag seen at a grazing angle) or a wrong size.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_cv2():
    """The system OpenCV (has aruco and a known API); falls back to whatever `import cv2` gives."""
    import importlib.machinery
    import importlib.util

    for path in glob.glob("/usr/lib/python3/dist-packages/cv2*.so"):
        loader = importlib.machinery.ExtensionFileLoader("cv2", path)
        spec = importlib.util.spec_from_file_location("cv2", path, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules["cv2"] = mod
        return mod
    import cv2

    return cv2


cv2 = load_cv2()


def make_detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers
    params = cv2.aruco.DetectorParameters_create()
    return lambda img: cv2.aruco.detectMarkers(img, d, parameters=params)


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return to_T(R.T, -R.T @ t)


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(sum(Rs))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def diagonal_intersection(q):
    """Image point where the quad's diagonals cross: the projection of a planar square's centre."""
    p1, p3, p2, p4 = (np.append(q[i], 1.0) for i in (0, 2, 1, 3))
    x = np.cross(np.cross(p1, p3), np.cross(p2, p4))
    return x[0] / x[2], x[1] / x[2]


def tag_depth(path, quad, scale, min_valid=0.5):
    """Median depth (m) over the tag's pixels from a saved aligned depth image, or None."""
    if not os.path.exists(path):
        return None
    depth = np.load(path)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    vals = depth[mask.astype(bool)]
    good = vals[vals > 0]
    if len(vals) == 0 or len(good) < min_valid * len(vals):
        return None  # mostly holes: the tag is too shiny or too oblique for the stereo depth
    return float(np.median(good)) * scale


def kabsch(P, Q):
    """R, t minimising |R P_i + t - Q_i|^2 over rows of P, Q (N, 3)."""
    pm, qm = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, qm - R @ pm


def solve_points(T_base_ee, p_cam, iters=200):
    """Translation-only eye-to-hand: the tag centre `o` is fixed in the hand frame, so
    T_base_ee @ o = T_base_cam @ p_cam for every pair. Unknowns: T_base_cam (6) and o (3).
    Alternates Kabsch (camera pose given o) with linear least squares (o given the camera pose).
    Uses the tag's position only, never its orientation, which a small tag gets wrong by
    tens of degrees when its pose is ambiguous."""
    Rb = np.array([T[:3, :3] for T in T_base_ee])
    tb = np.array([T[:3, 3] for T in T_base_ee])
    o = np.zeros(3)
    for _ in range(iters):
        R, t = kabsch(p_cam, np.einsum("nij,j->ni", Rb, o) + tb)
        q = (p_cam @ R.T + t) - tb  # = Rb @ o
        o_new = np.linalg.lstsq(Rb.reshape(-1, 3), q.reshape(-1), rcond=None)[0]
        if np.linalg.norm(o_new - o) < 1e-9:
            o = o_new
            break
        o = o_new
    R, t = kabsch(p_cam, np.einsum("nij,j->ni", Rb, o) + tb)
    err = np.linalg.norm(p_cam @ R.T + t - (np.einsum("nij,j->ni", Rb, o) + tb), axis=1) * 1e3
    return to_T(R, t), o, err


def residual(T_base_cam, T_base_ee, T_cam_tag):
    """Spread of the tag pose in the hand frame across pairs: per-pair translation (mm) and rotation (deg) errors."""
    T_ee_tag = [inv(Te) @ T_base_cam @ Tc for Te, Tc in zip(T_base_ee, T_cam_tag)]  # noqa: B905 (runs on the lab's Python 3.8: no strict=)
    t_mean = np.mean([T[:3, 3] for T in T_ee_tag], axis=0)
    R_mean = mean_rotation([T[:3, :3] for T in T_ee_tag])
    dt = np.array([np.linalg.norm(T[:3, 3] - t_mean) * 1e3 for T in T_ee_tag])
    dr = np.array([rot_deg(R_mean.T @ T[:3, :3]) for T in T_ee_tag])
    return dt, dr, to_T(R_mean, t_mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="the collector's --out directory")
    ap.add_argument("--marker-size", type=float, required=True, help="side of the black square, metres")
    ap.add_argument("--drop", type=int, default=0, help="drop this many worst pairs and re-solve")
    ap.add_argument("--no-depth", action="store_true", help="ignore saved depth; take the tag distance from its size")
    a = ap.parse_args()
    data = os.path.join(a.dir, "calib_data")
    meta = json.load(open(os.path.join(data, "meta.json")))
    K, D = np.load(os.path.join(data, "K.npy")), np.load(os.path.join(data, "D.npy"))
    detect = make_detector(meta["dict"])
    h = a.marker_size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    names, T_base_ee, T_cam_tag, n_depth = [], [], [], 0
    for img_path in sorted(glob.glob(os.path.join(data, "images", "*.png"))):
        stem = img_path[:-4]
        if not os.path.exists(stem + "_T_base_ee.npy"):
            continue
        corners, ids, _ = detect(cv2.imread(img_path))
        if ids is None or meta["marker_id"] not in ids.flatten():
            print(f"{os.path.basename(stem)}: tag not found, skipped")
            continue
        img_pts = corners[int(np.where(ids.flatten() == meta["marker_id"])[0][0])][0].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            print(f"{os.path.basename(stem)}: solvePnP failed, skipped")
            continue
        T_ct = to_T(cv2.Rodrigues(rvec)[0], tvec)
        z = tag_depth(stem + "_depth.npy", img_pts, meta.get("depth_scale", 0.001)) if not a.no_depth else None
        if z is not None:
            # Tag centre: the diagonals' intersection (projective centre), on its ray at the measured depth.
            # The tag is flat, so the median depth over the tag is its centre's depth.
            u, v = diagonal_intersection(img_pts)
            ray = np.linalg.solve(K, [u, v, 1.0])
            T_ct[:3, 3] = ray / ray[2] * z
        n_depth += z is not None
        names.append(os.path.basename(stem))
        T_base_ee.append(np.load(stem + "_T_base_ee.npy"))
        T_cam_tag.append(T_ct)
    n = len(names)
    print(f"tag distance from measured depth for {n_depth} of {n} pairs, from the tag's size for the rest")
    print(f"{n} usable pairs")
    if n < 5:
        raise SystemExit("need at least 5 pairs (15-20 for a good fit)")

    def opencv_compare(idx):
        """OpenCV's rotation-based solvers, printed for comparison only: they trust the tag's
        orientation, which flips on a small tag."""
        R_b2g = [inv(T_base_ee[i])[:3, :3] for i in idx]  # eye-to-hand: base->gripper in, camera->base out
        t_b2g = [inv(T_base_ee[i])[:3, 3] for i in idx]
        R_t2c = [T_cam_tag[i][:3, :3] for i in idx]
        t_t2c = [T_cam_tag[i][:3, 3] for i in idx]
        print("OpenCV hand-eye, for comparison (uses the tag's orientation):")
        for name in ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"):
            method = getattr(cv2, f"CALIB_HAND_EYE_{name}", None)
            if method is None:
                continue
            try:
                R, t = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
            except cv2.error:
                continue
            dt, _, _ = residual(to_T(R, t), [T_base_ee[i] for i in idx], [T_cam_tag[i] for i in idx])
            print(f"  {name:10s} tag-position rms {np.sqrt(np.mean(dt ** 2)):7.2f} mm")

    p_cam = np.array([T[:3, 3] for T in T_cam_tag])
    idx = list(range(n))
    T, o, err = solve_points([T_base_ee[i] for i in idx], p_cam[idx])
    for _ in range(a.drop):
        worst = idx[int(np.argmax(err))]
        print(f"dropping {names[worst]} ({err.max():.1f} mm)")
        idx.remove(worst)
        T, o, err = solve_points([T_base_ee[i] for i in idx], p_cam[idx])
    opencv_compare(idx)
    rms = float(np.sqrt(np.mean(err ** 2)))
    print(f"\ntranslation-only fit: {len(idx)} pairs, tag-position rms {rms:.2f} mm, max {err.max():.2f} mm")
    for name, e in sorted(zip([names[j] for j in idx], err), key=lambda x: -x[1])[:5]:  # noqa: B905
        print(f"  worst: {name}  {e:.1f} mm")
    print(f"camera position in base frame (m): {np.round(T[:3, 3], 4)}")
    print(f"tag centre in the hand frame (m): {np.round(o, 4)}")
    print("T_base_cam =\n" + np.array2string(T, precision=5, suppress_small=True))
    np.save(os.path.join(a.dir, "T_base_cam.npy"), T)
    with open(os.path.join(a.dir, "T_world_cam.json"), "w") as fh:
        json.dump({"T_world_cam": T.tolist(), "method": "eye-to-hand, tag positions only (Kabsch + hand offset)",
                   "pairs": len(idx), "rms_mm": rms, "max_mm": float(err.max()), "tag_in_hand_m": o.tolist(),
                   "marker_size_m": a.marker_size, **meta}, fh, indent=2)
    print(f"wrote {a.dir}/T_world_cam.json and T_base_cam.npy")


if __name__ == "__main__":
    main()
