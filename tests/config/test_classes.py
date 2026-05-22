import pytest

from deepreefmap.config.classes import DEFAULT_CLASSES_PATH, load_classes, read_classes_bytes


def test_load_coralscapes_classes_roles() -> None:
    classes_config = load_classes()

    assert classes_config.single_id_for_role("transect_line") == 15
    assert classes_config.single_id_for_role("transect_tools") == 8
    assert classes_config.single_id_for_role("background") == 13
    assert classes_config.ids_for_role("ignore_in_point_cloud") == {7, 8, 9, 13}
    assert classes_config.ids_for_role("ignore_in_cover") == {7, 8, 9, 13, 14}
    assert classes_config.name_for_id(5) == "sand"
    assert classes_config.color_for_id(5) == (194, 178, 128)


def test_cwd_classes_file_wins_over_bundled_copy(tmp_path, monkeypatch) -> None:
    # Expected behaviour: the default path is probed against the CWD before the
    # packaged resource, so a locally edited configs/classes_coralscapes.yaml is
    # honoured. read_classes_bytes must agree, or resume would key its cache on
    # bytes the run never used.
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "configs" / "classes_coralscapes.yaml"
    local.parent.mkdir()
    local.write_text('classes:\n  - {id: 1, name: "local only", color: [1, 2, 3], roles: []}\n')

    for path in (DEFAULT_CLASSES_PATH, None):
        assert load_classes(path).name_for_id(1) == "local only"
    assert read_classes_bytes(None) == local.read_bytes()


def test_bundled_classes_used_when_cwd_has_no_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_classes().name_for_id(5) == "sand"
    assert read_classes_bytes(None) == read_classes_bytes(DEFAULT_CLASSES_PATH)


def test_missing_custom_classes_path_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        load_classes(tmp_path / "nope.yaml")


def test_duplicate_ids_fail(tmp_path) -> None:
    path = tmp_path / "classes.yaml"
    path.write_text(
        "classes:\n"
        "  - {id: 1, name: sand, color: [1, 2, 3], roles: []}\n"
        "  - {id: 1, name: rubble, color: [3, 2, 1], roles: []}\n"
    )

    with pytest.raises(ValueError, match="Duplicate class id"):
        load_classes(path)
