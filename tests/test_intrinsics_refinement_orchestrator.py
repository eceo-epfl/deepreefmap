import numpy as np

from deepreefmap.pipeline.artifacts import MappingSequenceResult
from deepreefmap.pipeline.orchestrator import _mapping_without_world_points, _maybe_refine_intrinsics


class _RefiningMapper:
    def __init__(self, refined: np.ndarray | None):
        self._refined = refined

    def refine_intrinsics(self, mapping_result: MappingSequenceResult) -> np.ndarray | None:
        return self._refined


def _mapping_result() -> MappingSequenceResult:
    return MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 2, 2), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.ones((1, 2, 2, 3), dtype=np.float32),
    )


def test_maybe_refine_intrinsics_applies_mapper_override():
    mapping_result = _mapping_result()
    refined = np.array([[400.0, 0.0, 10.0], [0.0, 420.0, 11.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    out = _maybe_refine_intrinsics(
        mapping_name="loger",
        mapping=_RefiningMapper(refined),
        mapping_result=mapping_result,
        camera_profile_intrinsics=np.eye(3, dtype=np.float32),
        refine_intrinsics_from_mapper=True,
    )

    assert out.intrinsics[0, 0] == 400.0
    assert out.intrinsics[1, 1] == 420.0


def test_maybe_refine_intrinsics_keeps_original_when_mapper_returns_none():
    mapping_result = _mapping_result()

    out = _maybe_refine_intrinsics(
        mapping_name="scsfmlearner",
        mapping=_RefiningMapper(None),
        mapping_result=mapping_result,
        camera_profile_intrinsics=np.eye(3, dtype=np.float32),
        refine_intrinsics_from_mapper=True,
    )

    assert np.allclose(out.intrinsics, mapping_result.intrinsics)


def test_mapping_without_world_points_forces_unprojection_path():
    mapping_result = _mapping_result()

    out = _mapping_without_world_points(mapping_result)

    assert out.world_points is None
    assert np.allclose(out.depth_maps, mapping_result.depth_maps)
    assert np.allclose(out.intrinsics, mapping_result.intrinsics)
