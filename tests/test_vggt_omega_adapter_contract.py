import numpy as np
import pytest

from deepreefmap.mapping.vggt_omega_backend import (
    VGGTOmegaBackend,
    _balanced_target_shape,
    _confidence_from_depth_conf,
    _invert_extrinsics,
)


def test_vggt_omega_disables_per_frame_proxy_path():
    backend = VGGTOmegaBackend.__new__(VGGTOmegaBackend)
    with pytest.raises(RuntimeError, match="process_sequence"):
        backend.process_frame(0, np.zeros((4, 4, 3), dtype=np.uint8))


def test_constructor_rejects_resolution_not_multiple_of_patch():
    with pytest.raises(ValueError, match="multiple of 16"):
        VGGTOmegaBackend(image_resolution=500)


def test_balanced_target_shape_matches_upstream_for_processing_frames():
    # 1376x768 processing frames at resolution 512 -> 688x384 (w x h).
    aspect_ratio = 768 / 1376
    height, width = _balanced_target_shape(aspect_ratio, 512, 16)
    assert (width, height) == (688, 384)
    assert width % 16 == 0 and height % 16 == 0


def test_process_sequence_rejects_extreme_aspect_ratio():
    backend = VGGTOmegaBackend.__new__(VGGTOmegaBackend)
    backend._image_resolution = 512
    backend._torch = object()
    backend._model = object()
    # Aspect ratio H/W = 300/100 = 3.0 is outside [0.5, 2.0].
    tall_frame = np.zeros((300, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="aspect ratio"):
        backend.process_sequence([0], [tall_frame])


def test_invert_extrinsics_produces_world_to_camera_inverse():
    rotation = _rotation_z(0.4)
    translation = np.array([0.5, -0.2, 1.0])
    extr = np.zeros((1, 3, 4), dtype=np.float64)
    extr[0, :3, :3] = rotation
    extr[0, :3, 3] = translation

    poses_w_c = _invert_extrinsics(extr)

    extr_44 = np.eye(4)
    extr_44[:3, :4] = extr[0]
    assert np.allclose(poses_w_c[0] @ extr_44, np.eye(4), atol=1e-9)


def test_confidence_from_depth_conf_maps_into_unit_interval():
    depth_conf = np.array([1.0, 2.0, 10.0, 1e9], dtype=np.float32)
    conf = _confidence_from_depth_conf(depth_conf)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert conf[0] == pytest.approx(0.0)
    assert conf[1] == pytest.approx(0.5)
    assert conf[-1] == pytest.approx(1.0, abs=1e-6)


def test_predictions_to_result_packages_sequence_outputs():
    backend = _bare_backend()
    h, w = 4, 5
    frame_indices = [0, 1]
    extrinsics = np.stack([_identity_extrinsic(), _identity_extrinsic()], axis=0)
    extrinsics[1, :3, 3] = [0.1, 0.0, 0.0]  # second camera shifted in x
    pred_intrinsics = np.stack([_pinhole(w, h), _pinhole(w, h)], axis=0)
    depth = np.full((2, h, w), 3.0, dtype=np.float32)
    depth_conf = np.full((2, h, w), 4.0, dtype=np.float32)

    result = backend._predictions_to_result(
        frame_indices=frame_indices,
        extrinsics=extrinsics,
        pred_intrinsics=pred_intrinsics,
        depth=depth,
        depth_conf=depth_conf,
        target_size=(w, h),
    )

    assert result.depth_maps.shape == (2, h, w)
    assert result.poses_w_c.shape == (2, 4, 4)
    assert np.allclose(result.poses_w_c[0], np.eye(4), atol=1e-6)
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
    backend._image_size = (2 * w, 2 * h)
    pred_intrinsics = np.stack([_pinhole(w, h)], axis=0)

    result = backend._predictions_to_result(
        frame_indices=[0],
        extrinsics=np.stack([_identity_extrinsic()], axis=0),
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
    extrinsics = np.stack([_identity_extrinsic()], axis=0)
    pred_intrinsics = np.stack([_pinhole(w, h)], axis=0)
    depth = np.full((1, h, w), 3.0, dtype=np.float32)

    with pytest.raises(RuntimeError, match="refusing to silently drop frames"):
        backend._predictions_to_result(
            frame_indices=[0, 1, 2],
            extrinsics=extrinsics,
            pred_intrinsics=pred_intrinsics,
            depth=depth,
            depth_conf=None,
            target_size=(w, h),
        )


def test_refine_intrinsics_returns_median_predicted_k():
    backend = VGGTOmegaBackend.__new__(VGGTOmegaBackend)
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
    backend = VGGTOmegaBackend.__new__(VGGTOmegaBackend)
    backend._predicted_intrinsics = None
    assert backend.refine_intrinsics(mapping_result=None) is None


def _bare_backend() -> VGGTOmegaBackend:
    backend = VGGTOmegaBackend.__new__(VGGTOmegaBackend)
    backend._k = np.eye(3, dtype=np.float32)
    backend._image_size = None
    backend._predicted_intrinsics = None
    return backend


def _identity_extrinsic() -> np.ndarray:
    extr = np.zeros((3, 4), dtype=np.float64)
    extr[:3, :3] = np.eye(3)
    return extr


def _pinhole(w: int, h: int) -> np.ndarray:
    return np.array([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]], dtype=np.float32)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
