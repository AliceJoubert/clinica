from enum import Enum
from typing import Optional

from clinica.utils.pet import ReconstructionMethod, Tracer

__all__ = [
    "ADNIModality",
    "ADNIModalityConverter",
    "ADNIPETPreprocessingStep",
]


class ADNIModality(str, Enum):
    """Possible modalities supported by the ADNI-to-BIDS converter.

    These are the modalities exposed to the user. There is not a
    one-to-one relationship with the modality converters. That is,
    some modalities, like PET_AMYLOID, are associated with multiple
    converters, while others are associated with only one converter.
    For this reason, the ADNIModalityConverter enumeration exists,
    and there is a mapping between a ADNIModality and an iterable of
    ADNIModalityConverter variants.
    """

    T1 = "T1"
    PET_FDG = "PET_FDG"
    PET_AMYLOID = "PET_AMYLOID"
    PET_TAU = "PET_TAU"
    DWI = "DWI"
    FLAIR = "FLAIR"
    FMRI = "fMRI"
    FMAP = "FMAP"


class ADNIModalityConverter(str, Enum):
    """Possible modality converters for ADNI.

    These are not exposed to the user. However, there is a one-to-one
    relationship with the modality converters. That is, each variant
    has a corresponding converter.
    """

    T1 = "T1"
    PET_FDG = "PET_FDG"
    PET_FDG_8UNIFORM = "PET_FDG_8UNIFORM"
    PET_PIB = "PET_PIB"
    PET_FBB = "PET_FBB"
    PET_AV45 = "PET_AV45"
    PET_TAU = "PET_TAU"
    DWI = "DWI"
    FLAIR = "FLAIR"
    FMRI = "fMRI"
    FMAP = "FMAP"

    @property
    def is_pet(self) -> bool:
        return self in (
            ADNIModalityConverter.PET_FDG,
            ADNIModalityConverter.PET_FDG_8UNIFORM,
            ADNIModalityConverter.PET_PIB,
            ADNIModalityConverter.PET_AV45,
            ADNIModalityConverter.PET_TAU,
            ADNIModalityConverter.PET_FBB,
        )

    @property
    def output_folder(self) -> str:
        if self == ADNIModalityConverter.T1:
            return "anat"
        if self == ADNIModalityConverter.DWI:
            return "dwi"
        if self == ADNIModalityConverter.FLAIR:
            return "anat"
        if self == ADNIModalityConverter.FMRI:
            return "func"
        if self == ADNIModalityConverter.FMAP:
            return "fmap"
        if self.is_pet:
            return "pet"

    @property
    def json_sidecar(self) -> bool:
        if self == ADNIModalityConverter.T1:
            return False
        if self == ADNIModalityConverter.DWI:
            return True
        if self == ADNIModalityConverter.FLAIR:
            return True
        if self == ADNIModalityConverter.FMRI:
            return True
        if self == ADNIModalityConverter.FMAP:
            return True
        if self.is_pet:
            return False

    @property
    def to_center(self) -> bool:
        if self == ADNIModalityConverter.T1:
            return True
        if self == ADNIModalityConverter.DWI:
            return False
        if self == ADNIModalityConverter.FLAIR:
            return True
        if self == ADNIModalityConverter.FMRI:
            return False
        if self == ADNIModalityConverter.FMAP:
            return False
        if self.is_pet:
            return True


class ADNIPETPreprocessingStep(Enum):
    """ADNI preprocessing steps."""

    STEP0 = "ADNI Brain PET: Raw"
    STEP1 = "Co-registered Dynamic"
    STEP2 = "Co-registered, Averaged"
    STEP3 = "Coreg, Avg, Standardized Image and Voxel Size"
    STEP4_8MM = "Coreg, Avg, Std Img and Vox Siz, Uniform Resolution"
    STEP4_6MM = "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"

    @classmethod
    def list(cls) -> list:
        return list(ADNIPETPreprocessingStep)

    @classmethod
    def possibilities_inventory(cls) -> str:
        newline = "\n"
        return newline.join(
            [f"{cls.list().index(step)} : {step.value}" for step in cls.list()]
        )

    @classmethod
    def from_step_value(cls, step_value: int):
        """Accept step specification in raw integer (0, 1, ..., 5)."""
        error_msg = (
            f"Step value {step_value} is not a valid ADNI preprocessing step value."
            f"Valid values are : "
            f"{cls.possibilities_inventory()}."
        )
        if step_value != int(step_value):
            raise ValueError(error_msg)
        if 0 <= step_value <= 5:
            if step_value == 4:
                return cls.STEP4_8MM
            if step_value == 5:
                return cls.STEP4_6MM
            return cls[f"STEP{step_value}"]
        raise ValueError(error_msg)


def _get_output_filename(
    modality: ADNIModalityConverter, tracer: Optional[Tracer] = None
) -> str:
    # rq : tracer only defined for PET_AV45
    if modality == ADNIModalityConverter.T1:
        return "_T1w"
    if modality == ADNIModalityConverter.DWI:
        return "_dwi"
    if modality == ADNIModalityConverter.FLAIR:
        return "_FLAIR"
    if modality == ADNIModalityConverter.FMRI:
        return "_task-rest_bold"
    if modality == ADNIModalityConverter.FMAP:
        return "_fmap"
    if modality == ADNIModalityConverter.PET_FDG:
        return f"_trc-{Tracer.FDG.value}_rec-{ReconstructionMethod.CO_REGISTERED_AVERAGED.value}_pet"
    if modality == ADNIModalityConverter.PET_FDG_8UNIFORM:
        return f"_trc-{Tracer.FDG.value}_rec-{ReconstructionMethod.COREGISTERED_ISOTROPIC.value}_pet"
    if modality == ADNIModalityConverter.PET_PIB:
        return f"_trc-{Tracer.PIB.value}_pet"
    if modality == ADNIModalityConverter.PET_AV45:
        return f"_trc-{tracer.value}_pet"
    if modality == ADNIModalityConverter.PET_TAU:
        return f"_trc-{Tracer.AV1451.value}_pet"
