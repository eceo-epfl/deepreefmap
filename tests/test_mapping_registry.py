from unittest.mock import patch

from deepreefmap.mapping.lingbot_map_backend import LingBotMapBackend
from deepreefmap.mapping.loger_backend import LoGeRBackend
import pytest

from deepreefmap.mapping.registry import (
    backend_estimates_intrinsics,
    create_mapping_backend,
    list_mapping_backends,
)
from deepreefmap.mapping.scsfmlearner_backend import SCSfMLearnerBackend
from deepreefmap.mapping.vggt_omega_backend import VGGTOmegaBackend


def test_list_mapping_backends_includes_loger_star():
    names = list_mapping_backends()
    assert "scsfmlearner" in names
    assert "loger_star" in names
    assert "loger" in names


def test_list_mapping_backends_includes_vggt_omega():
    assert "vggt_omega" in list_mapping_backends()


def test_create_vggt_omega_backend_uses_registered_name():
    with patch.object(VGGTOmegaBackend, "_load_model", lambda self: None):
        backend = create_mapping_backend("vggt_omega")
    assert isinstance(backend, VGGTOmegaBackend)
    assert backend.name == "vggt_omega"
    assert backend._image_resolution == 512


def test_create_vggt_omega_backend_accepts_custom_resolution():
    with patch.object(VGGTOmegaBackend, "_load_model", lambda self: None):
        backend = create_mapping_backend("vggt_omega", image_resolution=640)
    assert backend._image_resolution == 640


def test_list_mapping_backends_includes_lingbot_map():
    assert "lingbot_map" in list_mapping_backends()


def test_create_lingbot_map_backend_uses_registered_name():
    with patch.object(LingBotMapBackend, "_load_model", lambda self: None):
        backend = create_mapping_backend("lingbot_map")
    assert isinstance(backend, LingBotMapBackend)
    assert backend.name == "lingbot_map"
    assert backend._mode == "streaming"
    assert backend._image_size == 518
    assert backend.default_window_size == 1


def test_create_lingbot_map_backend_accepts_windowed_options():
    with patch.object(LingBotMapBackend, "_load_model", lambda self: None):
        backend = create_mapping_backend(
            "lingbot_map", mode="windowed", window_size=32
        )
    assert backend._mode == "windowed"
    assert backend._window_size == 32
    assert backend.default_window_size == 32


def test_create_scsfmlearner_backend_uses_registered_name():
    with patch.object(SCSfMLearnerBackend, "_load_models", lambda self: None):
        backend = create_mapping_backend("scsfmlearner", checkpoint_path="dummy.pt")
    assert isinstance(backend, SCSfMLearnerBackend)
    assert backend.name == "scsfmlearner"
    assert backend._target_size == (512, 256)


def test_create_scsfmlearner_backend_accepts_custom_target_size():
    with patch.object(SCSfMLearnerBackend, "_load_models", lambda self: None):
        backend = create_mapping_backend(
            "scsfmlearner",
            checkpoint_path="dummy.pt",
            target_width=320,
            target_height=192,
        )
    assert isinstance(backend, SCSfMLearnerBackend)
    assert backend._target_size == (320, 192)


def test_backend_estimates_intrinsics_only_for_feed_forward_camera_models():
    assert backend_estimates_intrinsics("vggt_omega") is True
    assert backend_estimates_intrinsics("lingbot_map") is True
    assert backend_estimates_intrinsics("loger") is False
    assert backend_estimates_intrinsics("loger_star") is False
    assert backend_estimates_intrinsics("scsfmlearner") is False


def test_backend_estimates_intrinsics_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported mapping backend"):
        backend_estimates_intrinsics("nope")


def test_create_loger_star_uses_star_checkpoint_defaults():
    with patch.object(LoGeRBackend, "_load_loger", lambda self: None):
        backend = create_mapping_backend("loger_star")
    assert backend.name == "loger_star"
    assert "LoGeR_star" in backend._model_path
    assert "LoGeR_star" in backend._config_path
