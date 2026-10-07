"""Record a parity fixture (detections + appearance embeddings) from a MOT17 sequence.

The fixture feeds `tests/test_parity_mot17.py`: both trackers replay the same
real detections and embeddings, and must produce the same tracks.

Embeddings come from VisionCam's appearance embedder (MobileNetV2, 1280-d), on
the exact crops `deep_sort_realtime` would take, so this script runs from the
VisionCam root, in its environment:

    cd ~/projects/active/Project_VisionCam
    uv run python ~/projects/active/deepsort-rs/scripts/record_fixture.py \\
        --seq ~/datasets/MOT17/train/MOT17-04-SDP --frames 300 \\
        --out ~/projects/active/deepsort-rs/tests/fixtures/mot17-04-sdp.npz

To keep the fixture small enough for the repository, embeddings are projected
on their top `--dims` principal directions (uncentered SVD, the best rank-k
approximation of their dot products) and stored as float16. The tracker
normalizes its inputs, so the projection keeps real, correlated appearance
data; parity at full dimension is checked separately in VisionCam
(`tools.eval_mot`).
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())

from core.appearance_embedder import build_embedder  # noqa: E402 (VisionCam)
from core.tracker_backends import _embed  # noqa: E402 (VisionCam)


def load_public_detections(seq_dir):
    """det.txt -> {frame: [([l, t, w, h], conf, "person", None), ...]}, 0-based pixels."""
    by_frame = {}
    for row in np.loadtxt(os.path.join(seq_dir, "det", "det.txt"), delimiter=","):
        frame, _, left, top, width, height, conf = row[:7]
        if width <= 0 or height <= 0:
            continue
        by_frame.setdefault(int(frame), []).append(
            ([float(left) - 1, float(top) - 1, float(width), float(height)], float(conf), "person", None)
        )
    return by_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True, help="MOT17 sequence directory")
    parser.add_argument("--frames", type=int, default=0, help="number of frames, 0 = whole sequence")
    parser.add_argument("--dims", type=int, default=128, help="embedding dimensions kept")
    parser.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args()

    seq_dir = os.path.expanduser(args.seq)
    detections = load_public_detections(seq_dir)
    n_frames = args.frames or max(detections)
    embedder = build_embedder()

    frame_ids, ltwh, conf, embeddings = [], [], [], []
    for frame_id in range(1, n_frames + 1):
        image = cv2.imread(os.path.join(seq_dir, "img1", f"{frame_id:06d}.jpg"))
        if image is None:
            sys.exit(f"missing image for frame {frame_id}")
        dets, embeds = _embed(embedder, detections.get(frame_id, []), image)
        for (box, score, _, _), embed in zip(dets, embeds):
            frame_ids.append(frame_id)
            ltwh.append(box)
            conf.append(score)
            embeddings.append(np.asarray(embed, dtype=np.float32))

    full = np.stack(embeddings)
    _, _, vt = np.linalg.svd(full.astype(np.float64), full_matrices=False)
    projected = (full @ vt[: args.dims].T).astype(np.float16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        frame=np.asarray(frame_ids, dtype=np.int32),
        ltwh=np.asarray(ltwh, dtype=np.float32),
        conf=np.asarray(conf, dtype=np.float32),
        embedding=projected,
        n_frames=np.int32(n_frames),
        source=np.str_(os.path.basename(os.path.normpath(seq_dir))),
    )
    print(f"{args.out}: {n_frames} frames, {len(frame_ids)} detections, "
          f"embeddings {full.shape[1]}-d -> {args.dims}-d float16")


if __name__ == "__main__":
    main()
