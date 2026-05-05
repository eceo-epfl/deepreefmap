import json

import cv2
import numpy as np
import yaml

from deepreefmap.postproc.reports import (
    _build_legend,
    _present_class_ids,
    render_offline_video_placeholder,
)


def _write_classes_yaml(path):
    payload = {
        "classes": [
            {"id": 0, "name": "background", "color": [0, 0, 0], "roles": []},
            {"id": 1, "name": "coral", "color": [255, 0, 0], "roles": []},
            {"id": 2, "name": "sand", "color": [200, 200, 50], "roles": []},
        ]
    }
    path.write_text(yaml.safe_dump(payload))


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


def test_render_offline_video_4panel_layout_and_cumulative_ortho(tmp_path):
    h, w = 16, 24
    n_frames = 3

    frames = tmp_path / "frames"
    labels = tmp_path / "labels"
    frames.mkdir()
    labels.mkdir()

    classes_path = tmp_path / "classes.yaml"
    _write_classes_yaml(classes_path)

    frame_paths = []
    labels_paths = []
    for i in range(n_frames):
        fp = frames / f"{i:08d}.png"
        lp = labels / f"{i:08d}.npy"
        rgb = np.full((h, w, 3), i * 40, dtype=np.uint8)
        cv2.imwrite(str(fp), rgb)
        lab = np.full((h, w), i % 3, dtype=np.int32)
        np.save(lp, lab)
        frame_paths.append(f"frames/{i:08d}.png")
        labels_paths.append(f"labels/{i:08d}.npy")

    depths = np.linspace(0.5, 2.0, n_frames * h * w, dtype=np.float32).reshape(n_frames, h, w)
    np.savez_compressed(
        tmp_path / "mapping_outputs.npz",
        depth=depths,
        frame_indices=np.arange(n_frames, dtype=np.int32),
    )

    ortho_h, ortho_w = 32, 48
    ortho_rgb = np.full((ortho_h, ortho_w, 3), 100, dtype=np.uint8)
    ortho_labels = np.zeros((ortho_h, ortho_w), dtype=np.int32)
    ortho_labels[:, : ortho_w // 2] = 1
    ortho_labels[:, ortho_w // 2 :] = 2
    ortho_frame_index = np.full((ortho_h, ortho_w), -1, dtype=np.int32)
    ortho_frame_index[: ortho_h // 2, :] = 0
    ortho_frame_index[ortho_h // 2 :, : ortho_w // 2] = 1
    ortho_frame_index[ortho_h // 2 :, ortho_w // 2 :] = 2
    np.savez_compressed(
        tmp_path / "ortho.npz",
        rgb=ortho_rgb,
        labels=ortho_labels,
        frame_index=ortho_frame_index,
    )

    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "frame_paths": frame_paths,
                "labels_paths": labels_paths,
                "depth_maps": "mapping_outputs.npz",
                "classes": "classes.yaml",
            }
        )
    )

    render_offline_video_placeholder(tmp_path)

    out = tmp_path / "videos" / "qc_render.mp4"
    assert out.exists()

    cap = cv2.VideoCapture(str(out))
    try:
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    assert (vw, vh) == (w * 2, h * 2)
    assert n == n_frames


def test_present_class_ids_filters_to_classes_visible_under_mask():
    labels = np.array([[0, 1, 2], [1, 1, 7]], dtype=np.int32)
    valid = np.array([[True, True, False], [True, True, False]], dtype=bool)
    class_colors = {0: (0, 0, 0), 1: (10, 10, 10), 2: (20, 20, 20)}

    assert _present_class_ids(labels, valid, class_colors) == (0, 1)
    assert _present_class_ids(labels, np.zeros_like(valid), class_colors) == ()
    assert _present_class_ids(None, valid, class_colors) == ()


def test_build_legend_text_column_fits_longest_label():
    class_colors = {1: (10, 20, 30), 2: (40, 50, 60)}
    short = _build_legend(class_colors, {1: "a", 2: "b"}, target_h=200)
    long = _build_legend(class_colors, {1: "a", 2: "supercalifragilistic"}, target_h=200)

    assert long.shape[1] > short.shape[1]
    assert long.shape[0] == short.shape[0]
