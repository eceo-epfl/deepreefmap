from unittest.mock import patch

from deepreefmap.mapping.loger_backend import LoGeRBackend
from deepreefmap.mapping.registry import create_mapping_backend, list_mapping_backends
from deepreefmap.mapping.scsfmlearner_backend import SCSfMLearnerBackend


def test_list_mapping_backends_includes_loger_star():
    names = list_mapping_backends()
    assert "scsfmlearner" in names
    assert "loger_star" in names
    assert "loger" in names


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


def test_create_loger_star_uses_star_checkpoint_defaults():
    with patch.object(LoGeRBackend, "_load_loger", lambda self: None):
        backend = create_mapping_backend("loger_star")
    assert backend.name == "loger_star"
    assert "LoGeR_star" in backend._model_path
    assert "LoGeR_star" in backend._config_path
