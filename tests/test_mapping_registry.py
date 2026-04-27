from unittest.mock import patch

from deepreefmap.mapping.loger_backend import LoGeRBackend
from deepreefmap.mapping.registry import create_mapping_backend, list_mapping_backends


def test_list_mapping_backends_includes_loger_star():
    names = list_mapping_backends()
    assert "loger_star" in names
    assert "loger" in names


def test_create_loger_star_uses_star_checkpoint_defaults():
    with patch.object(LoGeRBackend, "_load_loger", lambda self: None):
        backend = create_mapping_backend("loger_star")
    assert backend.name == "loger_star"
    assert "LoGeR_star" in backend._model_path
    assert "LoGeR_star" in backend._config_path
