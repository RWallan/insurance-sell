from insurance_sell.modeling.train import Trainer
from insurance_sell.modeling.transformers import StringCleaner


def test_init_pipelines(test_model_settings):
    trainer = Trainer(test_model_settings)

    assert len(trainer._transformers) == 1
    assert trainer._transformers[0][0] == 'StringCleaner'
    assert isinstance(trainer._transformers[0][1], StringCleaner)


def test_init_model(test_model_settings):
    trainer = Trainer(test_model_settings)

    assert isinstance(
        trainer._model, test_model_settings.model_selection.model
    )
