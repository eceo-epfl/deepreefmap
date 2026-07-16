from deepreefmap.gui.run_history import distinct_model_combinations


def _row(mapping: str, segmentation: str) -> dict:
    return {"params": {"mapping_backend": mapping, "segmentation_model": segmentation}}


def test_distinct_combinations_dedup_preserves_newest_first_order() -> None:
    rows = [
        _row("loger_star", "coralscapes-vit-b-dpt"),
        _row("scsfmlearner", "coralscapes-vit-b-dpt"),
        _row("loger_star", "coralscapes-vit-b-dpt"),
        _row("scsfmlearner", "segformer-b2"),
    ]

    assert distinct_model_combinations(rows) == [
        ("loger_star", "coralscapes-vit-b-dpt"),
        ("scsfmlearner", "coralscapes-vit-b-dpt"),
        ("scsfmlearner", "segformer-b2"),
    ]


def test_distinct_combinations_empty() -> None:
    assert distinct_model_combinations([]) == []


def test_distinct_combinations_missing_params_become_none_pair() -> None:
    assert distinct_model_combinations([{"params": {}}]) == [(None, None)]
