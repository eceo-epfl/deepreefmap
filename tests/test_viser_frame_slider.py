from deepreefmap.visualization.viser_app import ViserLiveApp


def _app_without_init() -> ViserLiveApp:
    return ViserLiveApp.__new__(ViserLiveApp)


def test_normalize_slider_position_rounds_and_clamps() -> None:
    app = _app_without_init()
    assert app._normalize_slider_position(2.49, max_pos=5) == 2
    assert app._normalize_slider_position(2.51, max_pos=5) == 3
    assert app._normalize_slider_position(-10.0, max_pos=5) == 0
    assert app._normalize_slider_position(999.0, max_pos=5) == 5


def test_normalize_slider_position_handles_non_finite_inputs() -> None:
    app = _app_without_init()
    assert app._normalize_slider_position(float("nan"), max_pos=7) == 7
    assert app._normalize_slider_position(float("inf"), max_pos=7) == 7
    assert app._normalize_slider_position(-float("inf"), max_pos=7) == 7
    assert app._normalize_slider_position("not-a-number", max_pos=7) == 7
    assert app._normalize_slider_position(None, max_pos=7) == 7
