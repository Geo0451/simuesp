#!/usr/bin/env python3
"""DTW 1-NN baseline for air-written characters, with a live WebSocket feed for the
airtalk_live.html demo frontend.

    python airtalk_server.py                      # same offline eval as dtw_eval.py
    python airtalk_server.py --by session          # group by collection run instead of by day
    python airtalk_server.py --live --variant B    # recognise live + drive the frontend
    python airtalk_server.py --live --no-frontend  # plain console --live, no websocket server

Offline metrics are identical to dtw_eval.py (LOO / same-day / cross-day) and are reused
directly from it. What's new here is --live: it now also streams every raw packet to any
connected frontend (for the live "drawing" sketch) plus touch/char/caps events, in addition to
printing to the console exactly like dtw_eval.py did.

Requires `pip install websockets` for the frontend link (falls back to console-only --live
without it, same as before).
"""

import argparse
import sys
import time
from collections import deque

import numpy as np

from airpen_common import (
    DATA_DIR,
    PenStream,
    build_sample,
    derive,
    features,
    list_samples,
    load_sample,
    prepare_meta,
    vec,
)

# Reuse the offline eval + the DTW routine as-is -- no changes to that logic here.
from dtw_eval import dtw, evaluate, HAVE_NUMBA  # noqa: F401  (HAVE_NUMBA re-exported for parity)

try:
    import asyncio
    import json
    import threading

    import websockets

    HAVE_WEBSOCKETS = True
except ImportError:
    HAVE_WEBSOCKETS = False


# =============================================================================================
# WebSocket broadcaster -- runs its own asyncio loop on a background thread, so the existing
# synchronous PenStream loop doesn't need to change shape. `send()` is safe to call from the
# main thread at any point.
# =============================================================================================
class Broadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(5):
            print("[frontend] websocket server didn't come up in time; continuing without it.")

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except OSError as e:
            print(f"[frontend] couldn't start websocket server on {self.host}:{self.port}: {e}")
            self._ready.set()

    async def _serve(self):
        async def handler(ws, *_):
            self.clients.add(ws)
            try:
                async for _msg in ws:
                    pass  # frontend doesn't send anything back; just keep the socket open
            finally:
                self.clients.discard(ws)

        async with websockets.serve(handler, self.host, self.port):
            self._ready.set()
            await asyncio.Future()  # run forever

    def send(self, msg):
        if self.loop is None:
            return
        data = json.dumps(msg)
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)
        except RuntimeError:
            pass  # loop already shutting down

    async def _broadcast(self, data):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# =============================================================================================
# Packet-level capture loop.
#
# PenStream.next_touch() only returns once a whole touch (down -> release) has been captured,
# so there's no hook inside it to stream individual packets out as they arrive. This reimplements
# the same cut logic (identical thresholds/behaviour to airpen_common.PenStream.next_touch),
# reaching into the stream's own buffer/state, and broadcasts each packet plus a running 2D
# dead-reckoned trace (integrated gyro -> x,y) so the frontend can sketch the stroke live as it's
# written. The trace is for visual flavour only -- it's not what the recognizer uses; DTW below
# still runs on the same derived features as dtw_eval.py.
# =============================================================================================
def live_touch_with_stream(stream, broadcaster, release_ms=80.0, max_s=10.0, trace_scale=1.0):
    cap, start, last_on = None, 0, 0
    last_pkt = time.time()
    tx = ty = 0.0
    prev_t = None

    while True:
        p = stream._read()
        if p is None:
            if time.time() - last_pkt > 5:
                print("[stream] no packets for 5 s. Is the ESP32 on and on the same network?")
                last_pkt = time.time()
            continue
        last_pkt = time.time()

        rising = p[1] and not stream._prev_touch
        stream._prev_touch = p[1]

        if cap is None:
            stream.ring.append(p)
            while stream.ring[-1][0] - stream.ring[0][0] > stream.pre_s:
                stream.ring.popleft()
            if rising:
                cap = list(stream.ring)
                start = last_on = len(cap) - 1
                tx = ty = 0.0
                prev_t = cap[start][0]
                broadcaster.send({"type": "touch_start"})
            continue

        cap.append(p)
        t, touched, ax, ay, az, gx, gy, gz = p
        dt = max(0.0, t - prev_t) if prev_t is not None else 0.0
        prev_t = t
        if touched:
            last_on = len(cap) - 1
            tx += gy * dt * trace_scale
            ty += -gx * dt * trace_scale

        broadcaster.send({
            "type": "imu", "touched": bool(touched),
            "gx": gx, "gy": gy, "gz": gz,
            "x": tx, "y": ty,
        })

        if p[0] - cap[last_on][0] >= release_ms / 1000 or p[0] - cap[start][0] > max_s:
            tail = cap[-1][0]
            stream.ring = deque(q for q in cap if q[0] >= tail - stream.pre_s)
            broadcaster.send({"type": "touch_end"})
            return stream._pack(cap[:last_on + 1], start)


# =============================================================================================
# Live recognition (same tap/caps/DTW logic as dtw_eval.py, plus frontend events)
# =============================================================================================
TAP_S = 0.25
DOUBLE_TAP_GAP_S = 0.8


def live(args):
    v = args.variant.upper()
    paths = list_samples(args.data, args.user)
    if not paths:
        sys.exit(f"No templates in {args.data}/{args.user}. Run collect.py first.")
    X, y = [], []
    for p in paths:
        s = load_sample(p)
        d = derive(s)
        X.append(vec(features(d, v), d["fs"], args.len))
        y.append(s["label"])
    X, y = np.ascontiguousarray(np.stack(X)), np.array(y)
    band = max(1, int(round(args.band * args.len)))

    broadcaster = None
    if not args.no_frontend:
        if not HAVE_WEBSOCKETS:
            print("note: `pip install websockets` to drive the frontend; "
                  "continuing with console-only live mode.")
        else:
            broadcaster = Broadcaster(args.ws_host, args.ws_port)
            broadcaster.start()
            print(f"Frontend server listening on ws://{args.ws_host}:{args.ws_port} "
                  f"-- open airtalk_live.html, point it at that address, and connect.")

    stream = PenStream(args.port)
    meta = prepare_meta(stream, args.data, False, not args.no_rebias)
    caps = False
    pending_tap_t = None
    print(
        f"\nLoaded {len(y)} templates ({len(set(y))} classes), variant {v}. "
        f"Write a character, or double-tap to toggle caps. Ctrl+C to quit.\n"
    )
    print(f"  [caps: {'ON ' if caps else 'off'}]")
    if broadcaster:
        broadcaster.send({"type": "clear"})
        broadcaster.send({"type": "caps", "on": caps})

    try:
        while True:
            if broadcaster:
                cap = live_touch_with_stream(stream, broadcaster, release_ms=args.release_ms)
            else:
                cap = stream.next_touch(release_ms=args.release_ms)
            now = time.monotonic()

            if cap["duration"] < TAP_S:
                if (
                    pending_tap_t is not None
                    and now - pending_tap_t <= DOUBLE_TAP_GAP_S
                ):
                    caps = not caps
                    pending_tap_t = None
                    print(f"  [caps: {'ON ' if caps else 'off'}]")
                    if broadcaster:
                        broadcaster.send({"type": "caps", "on": caps})
                else:
                    pending_tap_t = now
                continue
            pending_tap_t = None

            if cap["duration"] < 0.4:
                continue
            d = derive(build_sample(cap, meta))
            x = vec(features(d, v), d["fs"], args.len)
            dist = np.array([dtw(x, t, band) for t in X])
            ranked = sorted((dist[y == c].min(), c) for c in set(y))
            best, second = ranked[0], ranked[1] if len(ranked) > 1 else (np.inf, "-")
            top3 = " | ".join(f"{c}: {dd:.1f}" for dd, c in ranked[:3])
            out = best[1].upper() if caps else best[1].lower()
            print(
                f"  -> {out}    (x{second[0] / max(best[0], 1e-9):.2f} margin)   [{top3}]"
            )
            if broadcaster:
                broadcaster.send({"type": "char", "char": out})
    except KeyboardInterrupt:
        print("\nbye")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--user", default="me")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--by", choices=["day", "session"], default="day")
    ap.add_argument(
        "--band",
        type=float,
        default=0.15,
        help="Sakoe-Chiba band as a fraction of length",
    )
    ap.add_argument("--len", type=int, default=100, help="resampled length")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--variant", default="B", help="feature variant for --live")
    ap.add_argument("--port", type=int, default=5005, help="UDP port for the pen")
    ap.add_argument("--no-rebias", action="store_true")
    ap.add_argument("--release-ms", type=float, default=80.0)
    ap.add_argument("--ws-host", default="0.0.0.0", help="websocket host for the frontend")
    ap.add_argument("--ws-port", type=int, default=8765, help="websocket port for the frontend")
    ap.add_argument(
        "--no-frontend",
        action="store_true",
        help="skip the websocket server; behaves like plain dtw_eval.py --live",
    )
    args = ap.parse_args()
    live(args) if args.live else evaluate(args)


if __name__ == "__main__":
    main()
