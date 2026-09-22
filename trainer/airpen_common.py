"""Shared helpers for the air-stylus tools: UDP stream, calibration, sample I/O, preprocessing.

Raw sensor data is what gets saved to disk. Everything derived (orientation, gravity removal,
heading alignment, features) is recomputed from raw at load time, so you can change the
preprocessing later (or add a magnetometer) without re-collecting anything.
"""
import glob
import json
import os
import socket
import struct
import time
from collections import deque
from datetime import datetime

import numpy as np

# ---- packet format: change this ONE line when the magnetometer arrives ---------------------
PKT_FMT = "<IB6f"  # u32 timestamp_us, u8 touched, ax ay az gx gy gz  (little-endian, 29 bytes)
PKT_SIZE = struct.calcsize(PKT_FMT)
DATA_DIR = "data"


# =============================================================================================
# UDP stream
# =============================================================================================
class PenStream:
    """Reads the ESP32 packets, keeps a rolling pre-touch buffer, and cuts out touch windows."""

    def __init__(self, port=5005, pre_s=1.0):
        self.pre_s = pre_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", port))
        except OSError as e:
            raise SystemExit(f"Can't bind UDP port {port}: {e}\n"
                             "Is the Godot visualizer (or another script) still running?")
        self.sock.settimeout(0.25)
        self.ring = deque()
        self._t = 0.0
        self._last_ts = None
        self._prev_touch = True  # conservative: require a release before the first touch-down
        self._warned = False

    # -- low level --------------------------------------------------------------------------
    def _parse(self, data):
        if len(data) != PKT_SIZE:
            if not self._warned:
                print(f"[stream] ignoring packet of {len(data)} bytes (expected {PKT_SIZE})")
                self._warned = True
            return None
        ts, touched, *vals = struct.unpack(PKT_FMT, data)
        if self._last_ts is not None:
            d = (ts - self._last_ts) & 0xFFFFFFFF  # survives u32 wraparound
            self._t += (d if d < 5_000_000 else 10_000) / 1e6  # huge jump = ESP reboot
        self._last_ts = ts
        return (self._t, bool(touched), *vals)

    def _read(self):
        try:
            data, _ = self.sock.recvfrom(2048)
        except socket.timeout:
            return None
        return self._parse(data)

    def drain(self):
        """Throw away whatever is queued in the OS socket buffer (stale data)."""
        self.sock.setblocking(False)
        try:
            while True:
                try:
                    data, _ = self.sock.recvfrom(2048)
                except (BlockingIOError, InterruptedError):
                    break
                self._parse(data)
        finally:
            self.sock.settimeout(0.25)
        self.ring.clear()
        self._prev_touch = True

    # -- fixed-duration capture (calibration) ------------------------------------------------
    def grab(self, seconds):
        """Return (N,6) raw [ax ay az gx gy gz] covering `seconds` of stream time."""
        out, t0, last = [], None, time.time()
        while True:
            p = self._read()
            if p is None:
                if time.time() - last > 5:
                    raise SystemExit("No packets for 5 s. Is the ESP32 on and on the same network?")
                continue
            last = time.time()
            t0 = p[0] if t0 is None else t0
            out.append(p[2:])
            if p[0] - t0 >= seconds:
                break
        self.ring.clear()
        self._prev_touch = True
        return np.array(out)

    def wait_still(self, seconds=1.5, gyro_std_max=2.0, accel_rel_std_max=0.03):
        """Block until the pen has been still for `seconds`; return that window."""
        self.drain()
        shown = False
        while True:
            a = self.grab(seconds)
            an = np.linalg.norm(a[:, :3], axis=1)
            if a[:, 3:].std(0).max() < gyro_std_max and an.std() / max(an.mean(), 1e-9) < accel_rel_std_max:
                return a
            if not shown:
                print("  ...waiting for the pen to be still")
                shown = True

    # -- touch capture -----------------------------------------------------------------------
    def next_touch(self, release_ms=80.0, max_s=10.0):
        """Block until one full touch (down -> up) has been captured.

        The touch flag is debounced: it must stay low for `release_ms` to count as released.
        Returns dict(t, raw, touch, start, end, duration). Data includes ~pre_s of pre-touch
        samples; [start:end] indexes the touched part.
        """
        cap, start, last_on = None, 0, 0
        last_pkt = time.time()
        while True:
            p = self._read()
            if p is None:
                if time.time() - last_pkt > 5:
                    print("[stream] no packets for 5 s. Is the ESP32 on and on the same network?")
                    last_pkt = time.time()
                continue
            last_pkt = time.time()
            rising = p[1] and not self._prev_touch
            self._prev_touch = p[1]
            if cap is None:
                self.ring.append(p)
                while self.ring[-1][0] - self.ring[0][0] > self.pre_s:
                    self.ring.popleft()
                if rising:
                    cap = list(self.ring)
                    start = last_on = len(cap) - 1
                continue
            cap.append(p)
            if p[1]:
                last_on = len(cap) - 1
            if p[0] - cap[last_on][0] >= release_ms / 1000 or p[0] - cap[start][0] > max_s:
                tail = cap[-1][0]
                self.ring = deque(q for q in cap if q[0] >= tail - self.pre_s)
                return self._pack(cap[:last_on + 1], start)

    @staticmethod
    def _pack(cap, start):
        a = np.array(cap, dtype=np.float64)  # cols: t, touch, ax ay az gx gy gz
        t = a[:, 0] - a[0, 0]
        return {"t": t, "raw": a[:, 2:].astype(np.float32), "touch": a[:, 1].astype(np.uint8),
                "start": start, "end": len(cap), "duration": float(t[-1] - t[start])}


# =============================================================================================
# Calibration (finds the pen axis for you, so you don't need to know how the IMU is mounted)
# =============================================================================================
def calib_path(data_dir):
    return os.path.join(data_dir, "calib.json")


def axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "XYZ"[i]


def run_calibration(stream, data_dir):
    print("\n=== One-time calibration ===")
    input("Step 1/3: lay the pen on the table and leave it. Press Enter... ")
    still = stream.wait_still(3.0)
    bias = still[:, 3:].mean(0)
    scale = float(np.linalg.norm(still[:, :3], axis=1).mean())
    print(f"  gyro bias (dps): {np.round(bias, 3)} | 1 g = {scale:.3f} raw accel units")

    input("Step 2/3: hold the pen. Press Enter, then twist it back and forth around its LONG axis\n"
          "         (like a screwdriver) for 5 seconds. ")
    stream.drain()
    print("  GO...")
    g = stream.grab(5.0)[:, 3:] - bias
    w, V = np.linalg.eigh(g.T @ g)
    axis, purity = V[:, -1], float(w[-1] / w.sum())
    print(f"  dominant rotation axis purity: {purity:.2f}")
    if purity < 0.6:
        print("  WARNING: rotation wasn't clean. Re-run with `--calibrate` and twist more purely.")

    input("Step 3/3: point the pen TIP at the ceiling and hold it still. Press Enter... ")
    still2 = stream.wait_still(1.5, gyro_std_max=4.0)
    d = float((still2[:, :3].mean(0) / scale) @ axis)
    if abs(d) < 0.7:
        print("  WARNING: pen doesn't look vertical. Re-run with `--calibrate`.")
    axis = axis * (1.0 if d >= 0 else -1.0)
    print(f"  pen axis (points to tip) in sensor frame: {np.round(axis, 3)}  (~{axis_name(axis)})")

    meta = {"pen_axis": axis.tolist(), "acc_scale": scale, "gyro_bias_dps": bias.tolist(),
            "created": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(data_dir, exist_ok=True)
    with open(calib_path(data_dir), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {calib_path(data_dir)}\n")
    return meta


def prepare_meta(stream, data_dir, force_calibrate=False, rebias=True):
    """Load calibration (running the wizard if needed) and re-zero the gyro bias for this session."""
    path = calib_path(data_dir)
    if force_calibrate or not os.path.exists(path):
        return run_calibration(stream, data_dir)  # already includes a fresh bias
    with open(path) as f:
        meta = json.load(f)
    if rebias:
        print("Set the pen down and keep it still for ~2 s (re-zeroing gyro)...")
        still = stream.wait_still(2.0)
        meta["gyro_bias_dps"] = still[:, 3:].mean(0).tolist()
        meta["acc_scale"] = float(np.linalg.norm(still[:, :3], axis=1).mean())
        print("  done.")
    return meta


# =============================================================================================
# Sample I/O
# =============================================================================================
def label_dir(label):
    """Folder name for a label (safe on case-insensitive filesystems)."""
    if len(label) == 1 and label.isascii() and label.isalnum():
        return label + "_lower" if label.islower() else label
    return "u" + "-".join(f"{ord(c):04x}" for c in label)


def build_sample(cap, meta, label="?", user="me", session=""):
    return dict(t=cap["t"], raw=cap["raw"], touch=cap["touch"], start=cap["start"], end=cap["end"],
                bias=np.asarray(meta["gyro_bias_dps"], float), acc_scale=float(meta["acc_scale"]),
                pen_axis=np.asarray(meta["pen_axis"], float), label=label, user=user, session=session)


def save_sample(data_dir, s):
    d = os.path.join(data_dir, s["user"], label_dir(s["label"]))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{s['session']}_{time.time_ns() // 1_000_000}.npz")
    np.savez_compressed(path, **s)
    return path


def load_sample(path):
    z = np.load(path, allow_pickle=False)
    s = {k: z[k] for k in z.files}
    for k in ("label", "user", "session"):
        s[k] = str(s[k])
    s["start"], s["end"], s["acc_scale"] = int(s["start"]), int(s["end"]), float(s["acc_scale"])
    s["path"] = path
    return s


def list_samples(data_dir, user):
    return sorted(glob.glob(os.path.join(data_dir, user, "*", "*.npz")))


def count_label(data_dir, user, label):
    return len(glob.glob(os.path.join(data_dir, user, label_dir(label), "*.npz")))


# =============================================================================================
# Orientation + preprocessing
# =============================================================================================
def tilt_quat(a):
    """Quaternion (w,x,y,z), body->earth, from one accel reading (yaw = 0)."""
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0, 0, 0])
    a = a / n
    z = np.array([0, 0, 1.0])
    w = 1.0 + a @ z
    if w < 1e-6:
        return np.array([0.0, 1, 0, 0])
    q = np.array([w, *np.cross(a, z)])
    return q / np.linalg.norm(q)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def madgwick_step(q, g, a, dt, beta):
    """One Madgwick IMU update. g in rad/s, a in any units (normalised inside). beta=0 -> gyro only."""
    w, x, y, z = q
    qdot = 0.5 * np.array([-x * g[0] - y * g[1] - z * g[2],
                           w * g[0] + y * g[2] - z * g[1],
                           w * g[1] - x * g[2] + z * g[0],
                           w * g[2] + x * g[1] - y * g[0]])
    n = np.linalg.norm(a)
    if beta > 0 and n > 1e-6:
        a = a / n
        f = np.array([2 * (x * z - w * y) - a[0],
                      2 * (w * x + y * z) - a[1],
                      2 * (0.5 - x * x - y * y) - a[2]])
        J = np.array([[-2 * y, 2 * z, -2 * w, 2 * x],
                      [2 * x, 2 * w, 2 * z, 2 * y],
                      [0.0, -4 * x, -4 * y, 0.0]])
        grad = J.T @ f
        gn = np.linalg.norm(grad)
        if gn > 0:
            qdot = qdot - beta * grad / gn
    q = q + qdot * dt
    return q / np.linalg.norm(q)


def derive(s, trim_ms=50.0, beta_free=0.15, beta_pen=0.02, accel_gate=0.25):
    """Raw sample -> dict of signals for the touch window (trimmed at both ends).

    Orientation is initialised from gravity at the start of the pre-touch buffer, then tracked by
    Madgwick: strong accel correction while the pen is up, very weak while it's touching (accel
    then contains real motion). Accel is also ignored whenever |a| is far from 1 g.
    Body frame: as measured.  World frame: z up, gravity removed from accel.
    Heading-aligned ('_h'): x = horizontal direction the pen tip points at touch-down.
    """
    t = s["t"].astype(float)
    N = len(t)
    dt = np.diff(t, prepend=t[0])
    fs = float(1.0 / np.median(dt[1:])) if N > 2 else 100.0
    acc = s["raw"][:, :3].astype(float) / s["acc_scale"]
    gyr = np.deg2rad(s["raw"][:, 3:].astype(float) - s["bias"])

    q = tilt_quat(acc[:min(5, N)].mean(0))
    Rs = np.empty((N, 3, 3))
    for i in range(N):
        touching = s["start"] <= i < s["end"]
        beta = beta_pen if touching else beta_free
        if abs(np.linalg.norm(acc[i]) - 1.0) > accel_gate:
            beta = 0.0
        q = madgwick_step(q, gyr[i], acc[i], dt[i], beta)
        Rs[i] = quat_to_R(q)

    k = int(round(trim_ms / 1000 * fs))
    lo, hi = s["start"] + k, s["end"] - k
    if hi - lo < 10:
        lo, hi = s["start"], s["end"]
    A, G, R = acc[lo:hi], gyr[lo:hi], Rs[lo:hi]
    lin_w = np.einsum("nij,nj->ni", R, A) - np.array([0, 0, 1.0])
    gyr_w = np.einsum("nij,nj->ni", R, G)

    p_w = np.einsum("nij,j->ni", R[:5], s["pen_axis"]).mean(0)
    heading_ok = bool(np.hypot(p_w[0], p_w[1]) > 0.3)  # pen not pointing (nearly) straight up/down
    h = float(np.arctan2(p_w[1], p_w[0])) if heading_ok else 0.0
    c, sn = np.cos(h), np.sin(h)

    def rot(v):
        return np.stack([c * v[:, 0] + sn * v[:, 1], -sn * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    return dict(fs=fs, acc_b=A, gyr_b=G, lin_w=lin_w, gyr_w=gyr_w,
                lin_h=rot(lin_w), gyr_h=rot(gyr_w), heading_ok=heading_ok)


VARIANTS = {
    "A": "raw body frame (6ch)",
    "B": "gravity-removed, world + heading aligned (6ch)",
    "C": "B + |accel|, |gyro| magnitudes (8ch)",
    "D": "yaw-invariant: z + horizontal magnitude (4ch)",
}


def _zh(v):  # (vertical, horizontal magnitude)
    return np.stack([v[:, 2], np.hypot(v[:, 0], v[:, 1])], 1)


def features(d, variant):
    """Returns a list of channel groups; each group is normalised separately in vec()."""
    if variant == "A":
        return [d["acc_b"], d["gyr_b"]]
    if variant == "B":
        return [d["lin_h"], d["gyr_h"]]
    if variant == "C":
        return [d["lin_h"], d["gyr_h"],
                np.linalg.norm(d["lin_h"], axis=1, keepdims=True),
                np.linalg.norm(d["gyr_h"], axis=1, keepdims=True)]
    if variant == "D":
        return [_zh(d["lin_w"]), _zh(d["gyr_w"])]
    raise ValueError(variant)


def _smooth(x, n):
    n = 2 * (n // 2) + 1
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    xp = np.pad(x, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(x.shape[1])], 1)


def vec(groups, fs, L=100):
    """Smooth (~40 ms), remove mean, divide each group by its own RMS (keeps the relative
    amplitude between axes inside a group), then resample to L steps."""
    cols = []
    for g in groups:
        g = _smooth(g, int(round(0.04 * fs)))
        g = g - g.mean(0)
        cols.append(g / (np.sqrt((g ** 2).mean()) + 1e-9))
    X = np.concatenate(cols, 1)
    src, dst = np.linspace(0, 1, len(X)), np.linspace(0, 1, L)
    return np.stack([np.interp(dst, src, X[:, c]) for c in range(X.shape[1])], 1)
