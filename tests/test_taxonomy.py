from pathlib import Path

import pytest

from deepreefmap.config.taxonomy import load_taxonomy


def test_load_coralscapes_taxonomy_roles():
    taxonomy = load_taxonomy(Path("configs/taxonomy_coralscapes.yaml"))

    assert taxonomy.single_id_for_role("transect_line") == 15
    assert taxonomy.single_id_for_role("transect_tools") == 8
    assert taxonomy.ids_for_role("ignore_in_point_cloud") == {0, 7, 8, 9, 13}
    assert taxonomy.ids_for_role("ignore_in_cover") == {0, 7, 8, 9, 13, 14}
    assert taxonomy.name_for_id(5) == "sand"


def test_duplicate_ids_fail(tmp_path):
    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        "classes:\n"
        "  - {id: 1, name: sand, roles: []}\n"
        "  - {id: 1, name: rubble, roles: []}\n"
    )

    with pytest.raises(ValueError, match="Duplicate taxonomy class id"):
        load_taxonomy(path)
