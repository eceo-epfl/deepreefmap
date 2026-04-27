from pathlib import Path

import pytest

from deepreefmap.config.classes import load_classes


def test_load_coralscapes_classes_roles():
    classes_config = load_classes(Path("configs/classes_coralscapes.yaml"))

    assert classes_config.single_id_for_role("transect_line") == 15
    assert classes_config.single_id_for_role("transect_tools") == 8
    assert classes_config.ids_for_role("ignore_in_point_cloud") == {7, 8, 9, 13}
    assert classes_config.ids_for_role("ignore_in_cover") == {7, 8, 9, 13, 14}
    assert classes_config.name_for_id(5) == "sand"
    assert classes_config.color_for_id(5) == (194, 178, 128)


def test_duplicate_ids_fail(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text(
        "classes:\n"
        "  - {id: 1, name: sand, color: [1, 2, 3], roles: []}\n"
        "  - {id: 1, name: rubble, color: [3, 2, 1], roles: []}\n"
    )

    with pytest.raises(ValueError, match="Duplicate class id"):
        load_classes(path)
