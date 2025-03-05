from pathlib import Path

from pydantic_settings import SettingsConfigDict

from insurance_sell.settings import (
    ModelParams,
    ModelSelectionParams,
    ModelSettings,
    Resampler,
    Transformer,
)


def test_model_settings_must_return_valid_model(monkeypatch):
    TEST_CONFIG_FILE = str(Path().cwd() / 'tests' / 'test_model_config.toml')
    monkeypatch.setattr(
        'insurance_sell.settings.ModelSettings.model_config',
        SettingsConfigDict(toml_file=TEST_CONFIG_FILE),
    )
    settings = ModelSettings()  # type: ignore

    assert isinstance(settings.model, ModelParams)
    assert isinstance(settings.model_selection, ModelSelectionParams)
    assert isinstance(settings.resampler, Resampler)
    assert len(settings.pipeline) == 1
    assert isinstance(settings.pipeline[0], Transformer)
    assert settings.features == ['Gender']
    assert settings.target == 'Result'
