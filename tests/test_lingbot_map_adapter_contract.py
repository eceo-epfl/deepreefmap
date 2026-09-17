import numpy as np
import pytest

from deepreefmap.mapping.lingbot_map_backend import (
    LingBotMapBackend,
    _auto_keyframe_interval,
    _crop_target_shape,
    _poses_w_c_from_decoded_pose_enc,
)
from deepreefmap.pipeline.artifacts import ReconstructionCancelled


def test_lingbot_map_disables_per_frame_proxy_path():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    with pytest.raises(RuntimeError, match="process_sequence"):
        backend.process_frame(0, np.zeros((4, 4, 3), dtype=np.uint8))


def test_constructor_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        LingBotMapBackend(mode="turbo")


def test_constructor_rejects_bad_attention():
    with pytest.raises(ValueError, match="attention must be one of"):
        LingBotMapBackend(attention="magic")


def test_constructor_rejects_image_size_not_multiple_of_patch():
    with pytest.raises(ValueError, match="multiple of 14"):
        LingBotMapBackend(image_size=500)


def test_crop_target_shape_matches_upstream_crop_mode():
    # 1376x768 processing frames at image_size 518 -> width 518, height 294.
    height, width = _crop_target_shape(768, 1376, 518, 14)
    assert (width, height) == (518, 294)
    assert width % 14 == 0 and height % 14 == 0


def test_crop_target_shape_refuses_portrait_frames():
    # Portrait input would center-crop the height upstream; we refuse instead.
    with pytest.raises(RuntimeError, match="centre-cropped"):
        _crop_target_shape(1376, 768, 518, 14)


def test_auto_keyframe_interval_streaming_short_and_long():
    assert _auto_keyframe_interval(100, "streaming") == 1
    assert _auto_keyframe_interval(320, "streaming") == 1
    assert _auto_keyframe_interval(1000, "streaming") == 4


def test_auto_keyframe_interval_windowed_is_one():
    assert _auto_keyframe_interval(1000, "windowed") == 1


def test_resolve_keyframe_interval_honors_explicit_override():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    backend._keyframe_interval = 5
    backend._mode = "streaming"
    assert backend._resolve_keyframe_interval(1000) == 5

    backend._keyframe_interval = None
    assert backend._resolve_keyframe_interval(1000) == 4


def test_decoded_pose_enc_is_used_as_camera_to_world_without_inversion():
    theta = 0.4
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    centre = np.array([0.5, -0.2, 1.0])
    decoded = np.zeros((1, 3, 4), dtype=np.float64)
    decoded[0, :3, :3] = rotation
    decoded[0, :3, 3] = centre

    poses_w_c = _poses_w_c_from_decoded_pose_enc(decoded)

    assert poses_w_c.shape == (1, 4, 4)
    assert np.allclose(poses_w_c[0, :3, :3], rotation)
    assert np.allclose(poses_w_c[0, :3, 3], centre)  # camera centre preserved, not -R^T t
    assert np.allclose(poses_w_c[0, 3], [0, 0, 0, 1])


def test_decoded_pose_enc_rejects_bad_shape():
    with pytest.raises(RuntimeError, match=r"\(S, 3, 4\)"):
        _poses_w_c_from_decoded_pose_enc(np.zeros((2, 3, 3)))


def test_predictions_to_result_packages_sequence_outputs():
    backend = _bare_backend()
    h, w = 4, 5
    frame_indices = [0, 1]
    poses_c2w = np.stack([_identity_pose(), _identity_pose()], axis=0)
    poses_c2w[1, :3, 3] = [0.1, 0.0, 0.0]  # second camera centre shifted +x in world
    pred_intrinsics = np.stack([_pinhole(w, h), _pinhole(w, h)], axis=0)
    depth = np.full((2, h, w), 3.0, dtype=np.float32)
    depth_conf = np.full((2, h, w), 4.0, dtype=np.float32)

    result = backend._predictions_to_result(
        frame_indices=frame_indices,
        poses_c2w=poses_c2w,
        pred_intrinsics=pred_intrinsics,
        depth=depth,
        depth_conf=depth_conf,
        target_size=(w, h),
    )

    assert result.depth_maps.shape == (2, h, w)
    assert result.poses_w_c.shape == (2, 4, 4)
    assert np.allclose(result.poses_w_c[0], np.eye(4), atol=1e-6)
    # Decoded pose_enc is camera-to-world already: the camera centre must come
    # through unchanged, not negated by a spurious inversion.
    assert np.allclose(result.poses_w_c[1, :3, 3], [0.1, 0.0, 0.0], atol=1e-6)
    # No pre-built points: the cloud stage must unproject depth with the single
    # K the orchestrator settles on, so geometry never carries a hidden K.
    assert result.world_points is None
    assert result.local_points is None
    assert result.confidence.shape == (2, h, w)
    assert result.confidence.min() >= 0.0 and result.confidence.max() <= 1.0
    assert result.scale_type == "relative"


def test_predictions_to_result_reports_calibrated_k_and_keeps_predicted_for_refine():
    backend = _bare_backend()
    h, w = 4, 5
    # Calibrated K at a 2x larger input frame; reported K must be rescaled to (w, h).
    backend._k = np.array([[200.0, 0, w], [0, 200.0, h], [0, 0, 1]], dtype=np.float32)
    backend._input_image_size = (2 * w, 2 * h)
    pred_intrinsics = np.stack([_pinhole(w, h)], axis=0)

    result = backend._predictions_to_result(
        frame_indices=[0],
        poses_c2w=np.stack([_identity_pose()], axis=0),
        pred_intrinsics=pred_intrinsics,
        depth=np.full((1, h, w), 3.0, dtype=np.float32),
        depth_conf=None,
        target_size=(w, h),
    )

    assert result.intrinsics[0, 0] == pytest.approx(100.0)
    assert result.intrinsics[0, 2] == pytest.approx(w / 2)
    assert result.intrinsics[1, 2] == pytest.approx(h / 2)
    assert np.allclose(backend._predicted_intrinsics, pred_intrinsics)
    assert np.allclose(backend.refine_intrinsics(mapping_result=None), pred_intrinsics[0])


def test_predictions_to_result_refuses_frame_count_mismatch():
    backend = _bare_backend()
    h, w = 4, 5
    poses_c2w = np.stack([_identity_pose()], axis=0)
    pred_intrinsics = np.stack([_pinhole(w, h)], axis=0)
    depth = np.full((1, h, w), 3.0, dtype=np.float32)

    with pytest.raises(RuntimeError, match="refusing to silently drop frames"):
        backend._predictions_to_result(
            frame_indices=[0, 1, 2],
            poses_c2w=poses_c2w,
            pred_intrinsics=pred_intrinsics,
            depth=depth,
            depth_conf=None,
            target_size=(w, h),
        )


def test_refine_intrinsics_returns_median_predicted_k():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    backend._predicted_intrinsics = np.stack(
        [
            np.array([[100.0, 0, 5.0], [0, 110.0, 4.0], [0, 0, 1]], dtype=np.float32),
            np.array([[120.0, 0, 5.0], [0, 130.0, 4.0], [0, 0, 1]], dtype=np.float32),
        ],
        axis=0,
    )

    refined = backend.refine_intrinsics(mapping_result=None)

    assert refined is not None
    assert refined[0, 0] == pytest.approx(110.0)
    assert refined[1, 1] == pytest.approx(120.0)
    assert refined[2].tolist() == [0.0, 0.0, 1.0]


def test_refine_intrinsics_returns_none_without_predictions():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    backend._predicted_intrinsics = None
    assert backend.refine_intrinsics(mapping_result=None) is None


def test_install_forward_progress_counts_frames_and_reports():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    model = _FakeModel()
    reports: list[tuple[int, int, str]] = []

    restore = backend._install_forward_progress(
        model, total_frames=10, cancel_event=None, progress_callback=lambda c, t, m: reports.append((c, t, m))
    )
    try:
        # A scale block of 8 frames, then two single streaming frames.
        model.forward(_FakeTensor((1, 8, 3, 4, 4)))
        model.forward(_FakeTensor((1, 1, 3, 4, 4)))
        model.forward(_FakeTensor((1, 1, 3, 4, 4)))
    finally:
        restore()

    assert [c for c, _, _ in reports] == [8, 9, 10]
    assert all(t == 10 for _, t, _ in reports)
    # After restore, forward no longer reports progress.
    model.forward(_FakeTensor((1, 1, 3, 4, 4)))
    assert len(reports) == 3


def test_install_forward_progress_raises_on_cancel():
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    model = _FakeModel()

    class _Event:
        def is_set(self):
            return True

    restore = backend._install_forward_progress(
        model, total_frames=10, cancel_event=_Event(), progress_callback=None
    )
    try:
        with pytest.raises(ReconstructionCancelled):
            model.forward(_FakeTensor((1, 8, 3, 4, 4)))
    finally:
        restore()
    # Cancellation must short-circuit before the underlying forward runs.
    assert model.calls == 0


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def forward(self, images, *args, **kwargs):
        self.calls += 1
        return {"pose_enc": images}


def _bare_backend() -> LingBotMapBackend:
    backend = LingBotMapBackend.__new__(LingBotMapBackend)
    backend._k = np.eye(3, dtype=np.float32)
    backend._input_image_size = None
    backend._predicted_intrinsics = None
    return backend


def _identity_pose() -> np.ndarray:
    pose = np.zeros((3, 4), dtype=np.float64)
    pose[:3, :3] = np.eye(3)
    return pose


def _pinhole(w: int, h: int) -> np.ndarray:
    return np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]], dtype=np.float32)
