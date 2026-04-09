from enum import Enum

import pandas as pd

from clinica.converters.adni_to_bids._utils import ADNIModalityConverter
from clinica.utils.stream import cprint

__all__ = [
    "get_images_pet",
    "ADNIPETPreprocessingStep",
    "ADNITracer",
    "define_pet_processing_step_with_tracer",
]


class ADNITracer(str, Enum):
    AV45 = "AV45"
    FBB = "FBB"
    PIB = "PIB"
    FDG = "FDG"
    AV1451 = "AV1451"


class ADNIPETPreprocessingStep(Enum):
    """ADNI preprocessing steps."""

    STEP0 = "ADNI Brain PET: Raw"
    STEP1 = "Co-registered Dynamic"
    STEP2 = "Co-registered, Averaged"
    STEP3 = "Coreg, Avg, Standardized Image and Voxel Size"
    STEP4_8MM = "Coreg, Avg, Std Img and Vox Siz, Uniform Resolution"
    STEP4_6MM = "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"

    @classmethod
    def from_step_value(cls, step_value: int):
        """Accept step specification in raw integer (0, 1, ..., 5)."""
        error_msg = (
            f"Step value {step_value} is not a valid ADNI preprocessing step value."
            f"Valid values are : \n"
            f"{"\n".join([f"{step.index} : {step}" for step in list(ADNIPETPreprocessingStep)])}."
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


def define_pet_processing_step_with_tracer(
    tracer: ADNITracer, step: ADNIPETPreprocessingStep
) -> str:
    if step == ADNIPETPreprocessingStep.STEP0:
        return f"{step.value} {tracer.value}"
    if tracer == ADNITracer.FDG:
        return step.value
    return f"{tracer.value} {step.value}"


def _get_modality_from_adni_preprocessing_step(
    tracer: ADNITracer,
    step: ADNIPETPreprocessingStep,
) -> ADNIModalityConverter:
    if tracer == ADNITracer.FDG:
        if step == ADNIPETPreprocessingStep.STEP2:
            return ADNIModalityConverter.PET_FDG
        if step == ADNIPETPreprocessingStep.STEP4_8MM:
            return ADNIModalityConverter.PET_FDG_UNIFORM
    raise ValueError(
        f"The ADNI preprocessing step {step} is not (yet) supported by the converter for PET tracer {tracer}."
        f"The converter only supports {ADNIPETPreprocessingStep.STEP2} and "
        f"{ADNIPETPreprocessingStep.STEP4_8MM} for now."
    )


def get_images_pet(
    subject: str,
    pet_qc_subj: pd.DataFrame,
    subject_pet_meta: pd.DataFrame,
    df_cols: list[str],
    modality: str,
    sequences_preprocessing_step: list[str],
    viscode_field: str = "VISCODE2",
) -> list[pd.DataFrame]:
    """Selection of scans passing QC and at the chosen preprocessing stage is performed.

    Args:
        subject: Subject identifier
        pet_qc_subj: Dataframe containing QC for scans for the subject
        subject_pet_meta: Dataframe containing metadata for scans for the subject
        df_cols: Columns of output dataframe
        modality: Imaging modality
        sequences_preprocessing_step: List of sequence names that correspond to desired preprocessing stage
        viscode_field: Name of the field in the pet_qc_subj dataframe that provides to the visit code

    Returns: Dataframe containing images metadata
    """
    from clinica.utils.filemanip import replace_special_characters_with_symbol
    from clinica.utils.pet import Tracer

    subj_dfs_list = []
    for visit in list(pet_qc_subj[viscode_field].unique()):
        if pd.isna(visit):
            continue
        pet_qc_visit = pet_qc_subj[pet_qc_subj[viscode_field] == visit]
        if pet_qc_visit.empty:
            continue
        # If there are several scans for a timepoint we start with image acquired last (higher LONIUID)
        pet_qc_visit = pet_qc_visit.sort_values("LONIUID", ascending=False)

        original_pet_meta = pd.DataFrame(columns=subject_pet_meta.columns)
        qc_visit = pet_qc_visit.iloc[0]
        for qc_index in range(len(pet_qc_visit)):
            qc_visit = pet_qc_visit.iloc[qc_index]

            # We are looking for FDG PET metadata of Original images, that passed QC,
            # acquired at the same date as the current scan that passed QC for the current visit,
            # not containing ‘early’ in the sequence name

            original_pet_meta = subject_pet_meta[
                (subject_pet_meta["Orig/Proc"] == "Original")
                & (subject_pet_meta["Image ID"] == int(qc_visit.LONIUID[1:]))
                & (subject_pet_meta["Scan Date"] == qc_visit.EXAMDATE)
                & ~subject_pet_meta.Sequence.str.contains("early", case=False, na=False)
            ]
            # Check if we found a matching image. If yes, we stop looking for it.
            if not original_pet_meta.empty:
                break

        if original_pet_meta.empty:
            cprint(
                f"No {modality} images metadata for subject {subject} and visit {qc_visit[viscode_field]}",
                lvl="info",
            )
            continue

        original_image = original_pet_meta.iloc[0]

        # Co-registered and Averaged image with the same Series ID of the original image
        averaged_pet_meta = subject_pet_meta[
            subject_pet_meta["Sequence"].isin(sequences_preprocessing_step)
            & (subject_pet_meta["Series ID"] == original_image["Series ID"])
        ]

        # If an explicit Co-registered, Averaged image does not exist,
        # the original image is already in that preprocessing stage.

        if averaged_pet_meta.empty:
            sel_image = original_image
            original = True
        else:
            sel_image = averaged_pet_meta.iloc[0]
            original = False

        phase = "ADNI1" if modality == "PIB-PET" else qc_visit.Phase
        visit = sel_image.Visit
        sequence = replace_special_characters_with_symbol(
            sel_image.Sequence, symbol="_"
        )
        date = sel_image["Scan Date"]
        study_id = sel_image["Study ID"]
        series_id = sel_image["Series ID"]
        image_id = sel_image["Image ID"]

        # If it is an amyloid PET we need to find which is the tracer of the scan and add it to the
        if modality == "Amyloid-PET":
            if "av45" in sel_image.Sequence.lower():
                tracer = Tracer.AV45.value
            elif "fbb" in sel_image.Sequence.lower():
                tracer = Tracer.FBB.value
            else:
                cprint(
                    msg=(
                        f"Unknown tracer for Amyloid PET image metadata for subject {subject} "
                        f"for visit {qc_visit[viscode_field]}"
                    ),
                    lvl="warning",
                )
                continue

            scan_data = [
                [
                    phase,
                    subject,
                    qc_visit[viscode_field],
                    str(visit),
                    sequence,
                    date,
                    str(study_id),
                    str(series_id),
                    str(image_id),
                    original,
                    tracer,
                ]
            ]
        else:
            scan_data = [
                [
                    phase,
                    subject,
                    qc_visit[viscode_field],
                    str(visit),
                    sequence,
                    date,
                    str(study_id),
                    str(series_id),
                    str(image_id),
                    original,
                ]
            ]

        row_to_append = pd.DataFrame(scan_data, columns=df_cols)
        subj_dfs_list.append(row_to_append)

    return subj_dfs_list
