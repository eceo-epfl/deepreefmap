import json

import cv2
import numpy as np

from deepreefmap.postproc.reports import render_offline_video_placeholder


def test_render_offline_video_uses_manifest(tmp_path):
    frames = tmp_path / "frames"
    labels = tmp_path / "labels"
    frames.mkdir()
    labels.mkdir()
    frame_path = frames / "00000000.png"
    labels_path = labels / "00000000.npy"
    cv2.imwrite(str(frame_path), np.zeros((8, 8, 3), dtype=np.uint8))
    np.save(labels_path, np.ones((8, 8), dtype=np.int32))
    np.savez_compressed(tmp_path / "mapping_outputs.npz", depth=np.ones((1, 8, 8), dtype=np.float32))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "frame_paths": ["frames/00000000.png"],
                "labels_paths": ["labels/00000000.npy"],
                "depth_maps": "mapping_outputs.npz",
            }
        )
    )

    render_offline_video_placeholder(tmp_path)

    assert (tmp_path / "videos" / "qc_render.mp4").exists()
