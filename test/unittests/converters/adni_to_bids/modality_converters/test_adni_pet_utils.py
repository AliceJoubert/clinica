import pytest

from clinica.converters.adni_to_bids._utils import (
    ADNIModalityConverter,
    ADNIPETPreprocessingStep,
)
from clinica.converters.adni_to_bids.modality_converters._pet_utils import (
    ADNITracer,
)


@pytest.mark.parametrize(
    "tracer, step, expected",
    [
        (ADNITracer.PIB, ADNIPETPreprocessingStep.STEP0, "ADNI Brain PET: Raw PIB"),
        (ADNITracer.FDG, ADNIPETPreprocessingStep.STEP0, "ADNI Brain PET: Raw FDG"),
        (ADNITracer.FDG, ADNIPETPreprocessingStep.STEP1, "Co-registered Dynamic"),
        (ADNITracer.PIB, ADNIPETPreprocessingStep.STEP2, "PIB Co-registered, Averaged"),
    ],
)
def test_define_pet_processing_step_with_tracer(tracer, step, expected):
    from clinica.converters.adni_to_bids.modality_converters._pet_utils import (
        define_pet_processing_step_with_tracer,
    )

    assert expected == define_pet_processing_step_with_tracer(tracer, step)


@pytest.mark.parametrize(
    "step_value,tracer,expected",
    [
        (2, ADNIModalityConverter.PET_FDG, ADNIModalityConverter.PET_FDG),
        (4, ADNIModalityConverter.PET_FDG, ADNIModalityConverter.PET_FDG_UNIFORM),
    ],
)
def test_get_modality_from_adni_preprocessing_step(step_value, tracer, expected):
    from clinica.converters.adni_to_bids.modality_converters._pet_utils import (
        _check_modality_with_preprocessing_step,
    )

    assert (
        _check_modality_with_preprocessing_step(
            tracer, ADNIPETPreprocessingStep.from_step_value(step_value)
        )
        == expected
    )


@pytest.mark.parametrize("step_value", [0, 1, 3, 5])
def test_get_modality_from_adni_preprocessing_step_wrong_value(step_value):
    from clinica.converters.adni_to_bids.modality_converters._pet_utils import (
        _check_modality_with_preprocessing_step,
    )

    with pytest.raises(
        ValueError,
        match="The ADNI preprocessing step",
    ):
        _check_modality_with_preprocessing_step(
            ADNIModalityConverter.PET_FDG,
            ADNIPETPreprocessingStep.from_step_value(step_value),
        )


def test_get_modality_from_adni_preprocessing_step_unsupported_modality():
    from clinica.converters.adni_to_bids.modality_converters._pet_utils import (
        _check_modality_with_preprocessing_step,
    )

    with pytest.raises(
        ValueError,
        match="The ADNI preprocessing step",
    ):
        _check_modality_with_preprocessing_step(
            ADNIModalityConverter.PET_PIB,
            ADNIPETPreprocessingStep.from_step_value(3),
        )
