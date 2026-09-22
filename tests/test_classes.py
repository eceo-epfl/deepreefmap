from pathlib import Path

import pytest

from deepreefmap.config.classes import classes_path_exists, load_classes


def test_load_coralscapes_classes_roles():
    classes_config = load_classes(Path("configs/classes_coralscapes.yaml"))

    assert classes_config.single_id_for_role("transect_line") == 15
    assert classes_config.single_id_for_role("transect_tools") == 8
    assert classes_config.ids_for_role("ignore_in_point_cloud") == {7, 8, 9, 13}
    assert classes_config.ids_for_role("ignore_in_cover") == {7, 8, 9, 13, 14}
    assert classes_config.name_for_id(5) == "sand"
    assert classes_config.color_for_id(5) == (194, 178, 128)


def test_load_coralscapesv2_95_classes_roles():
    classes_config = load_classes(Path("configs/classes_coralscapesv2_95.yaml"))

    assert len(classes_config.classes) == 95
    # Ids are contiguous 1..95 (id 0 = unlabeled is dropped, matching V1 convention).
    assert sorted(cls.id for cls in classes_config.classes) == list(range(1, 96))
    assert classes_config.single_id_for_role("transect_line") == 88
    assert classes_config.single_id_for_role("transect_tools") == 89
    assert classes_config.ids_for_role("ignore_in_point_cloud") == {14, 25, 39, 89, 93, 94}
    assert classes_config.ids_for_role("ignore_in_cover") == {14, 22, 25, 39, 89, 93, 94}
    assert classes_config.name_for_id(25) == "fish"
    assert classes_config.name_for_id(69) == "sand"
    assert classes_config.color_for_id(69) == (194, 178, 128)


def test_packaged_resource_fallback(monkeypatch, tmp_path):
    # From a directory without the repo-relative configs/, a relative classes
    # path must still resolve through the packaged deepreefmap.resources copy.
    monkeypatch.chdir(tmp_path)
    rel = Path("configs/classes_coralscapesv2_95.yaml")
    assert not rel.exists()
    assert classes_path_exists(rel)
    classes_config = load_classes(rel)
    assert len(classes_config.classes) == 95


def test_duplicate_ids_fail(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text(
        "classes:\n"
        "  - {id: 1, name: sand, color: [1, 2, 3], roles: []}\n"
        "  - {id: 1, name: rubble, color: [3, 2, 1], roles: []}\n"
    )

    with pytest.raises(ValueError, match="Duplicate class id"):
        load_classes(path)
