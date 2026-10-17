import pytest
from pydantic import ValidationError

from janus.core.config import MarketConfig, SimulationConfig


def test_configs_are_strict_and_immutable() -> None:
    with pytest.raises(ValidationError):
        MarketConfig.model_validate({"reference_price": 100, "mystery": 3})
    config = SimulationConfig()
    with pytest.raises(ValidationError):
        config.root_seed = 5  # type: ignore[misc]


def test_decision_period_is_derived_from_timing() -> None:
    assert SimulationConfig(pre_shock_periods=20, observation_periods=10).decision_period == 30
