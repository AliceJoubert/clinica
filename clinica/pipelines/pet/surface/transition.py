from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib

from clinica.pipelines.utils import FreeSurferAnnotationImage
from clinica.utils.image import HemiSphere


def reformat_surfname(
    hemisphere: HemiSphere, left_surface: Path, right_surface: Path
) -> Path:
    if hemisphere == HemiSphere.LEFT:
        return left_surface
    if hemisphere == HemiSphere.RIGHT:
        return right_surface


def project_onto_fsaverage(
    projection: Path,
    subject_id: str,
    session_id: str,
    caps_dir: Path,
    fwhm: int,
    is_longitudinal: bool,
    output_dir: Optional[Path] = None,
) -> Path:
    """Project data onto an averaged subject called fsaverage.

    This subject is available in the $SUBJECTS_DIR folder.

    Notes
    -----
    fsaverage and the subject must be in the subject_dir, so a copy of fsaverage is performed if necessary.

    Parameters
    ----------
    projection : Path
        The path to the projected data onto native subject surface.

    subject_id : str
        The subject id (something like sub-ADNI002S4213).

    session_id : str
        The session id ( something like : ses-M012).

    caps_dir : Path
        The path to the CAPS directory.

    fwhm : int
        FWHM of the Gaussian filter used for smoothing on fsaverage surface (not volume !)

    is_longitudinal : bool
        Longitudinal pipeline or not.

    output_dir : Path, optional
        The path to the output folder in which to write the output files.
        If not provided, the files will be written in the current directory.

    Returns
    -------
    Path :
        The path to the data averaged.
    """
    import os
    import shutil

    subjects_directory_backup = Path(os.path.expandvars("$SUBJECTS_DIR"))
    subjects_directory, freesurfer_id = (
        _get_new_subjects_directory_longitudinal
        if is_longitudinal
        else _get_new_subjects_directory
    )(caps_dir, subject_id, session_id)
    os.environ["SUBJECTS_DIR"] = str(subjects_directory)
    # copy fsaverage folder next to : subject_id + '_' + session_id
    # for the mris_preproc command to properly find src and target
    fsaverage_has_been_copied = False
    if not (subjects_directory / "fsaverage").exists():
        shutil.copytree(
            str(subjects_directory_backup / "fsaverage"),
            str(subjects_directory / "fsaverage"),
        )
        fsaverage_has_been_copied = True
    # also copy the mgh file in the surf folder (needed by MRISPreproc
    projection_in_surf_folder = (
        subjects_directory / freesurfer_id / "surf" / projection.name
    )
    if not projection_in_surf_folder.exists():
        shutil.copy(str(projection), str(projection_in_surf_folder))
    out_fsaverage = (
        output_dir or Path.cwd()
    ) / f"fsaverage_fwhm-{fwhm}_{projection.name}"
    _run_mris_preproc_as_standalone_nipype_node(
        projection, freesurfer_id, fwhm, out_fsaverage
    )
    # remove projection file from surf folder
    projection_in_surf_folder.unlink(missing_ok=False)
    # remove fsaverage if it has been copied
    if fsaverage_has_been_copied:
        shutil.rmtree(subjects_directory / "fsaverage")
    # put back original subjects_dir env
    os.environ["SUBJECTS_DIR"] = str(subjects_directory_backup)
    return out_fsaverage


def _run_mris_preproc_as_standalone_nipype_node(
    projection: Path,
    freesurfer_id: str,
    fwhm: float,
    output_file: Path,
):
    from nipype.interfaces.freesurfer import MRISPreproc

    projection_node = MRISPreproc()
    projection_node.inputs.target = "fsaverage"
    projection_node.inputs.subjects = [freesurfer_id]
    projection_node.inputs.fwhm = fwhm
    projection_node.inputs.hemi = HemiSphere(projection.name[0:2]).value
    projection_node.inputs.surf_measure = projection.name[3:]
    projection_node.inputs.out_file = str(output_file)
    projection_node.run()


def compute_average_pet_signal_based_on_annotations(
    pet_projections: Tuple[Path, Path],
    atlas_files: Dict[str, FreeSurferAnnotationImage],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Computes the average of PET signal based on annot files from Freesurfer.

    Those files describe the brain according to known atlases.

    Parameters
    ----------
    pet_projections : tuple of two Path
        The paths to the PET projection (must be a MGH file) [left_hemisphere, right_hemisphere].

    atlas_files : dict[str, FreeSurferAnnotationImage]
        Dictionary containing path to lh and rh annotation files for any number of atlases.

    output_dir : Path, optional
        The path to the output folder in which to write the output files.
        If not provided, the files will be written in the current directory.

    Returns
    -------
    Path :
        The path to the tsv containing average PET values.

    Raises
    ------
    ValueError :
        If not exactly two files were provided through the argument 'pet_projections'.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd

    from clinica.pipelines.utils import FreeSurferAnnotation
    from clinica.utils.stream import log_and_raise

    if len(pet_projections) != 2:
        msg = (
            "The compute_average_pet_signal_based_on_annotations function requires two files "
            "for the argument 'pet_projections', one for the left hemisphere, one for the right. "
            f"The following {len(pet_projections)} were received:\n"
            + "\n".join([str(_) for _ in pet_projections])
        )
        log_and_raise(msg, ValueError)
    pet_mgh = {
        HemiSphere.LEFT: np.squeeze(
            nib.load(pet_projections[0]).get_fdata(dtype="float32")
        ),
        HemiSphere.RIGHT: np.squeeze(
            nib.load(pet_projections[1]).get_fdata(dtype="float32")
        ),
    }
    filename_tsv = []
    for atlas_name, annotation_image in atlas_files.items():
        annotation = FreeSurferAnnotation.from_annotation_image(annotation_image)
        annotation.replace_minus_one_annotation_with_zero()
        average_region = []
        for region_id, region_name in enumerate(annotation.region_names):
            for hemisphere in (HemiSphere.LEFT, HemiSphere.RIGHT):
                mask = annotation.get_annotation(hemisphere) == region_id
                mask = np.uint(mask)
                masked_data = mask * pet_mgh[hemisphere]
                average_region.append(
                    np.nan if np.sum(mask) == 0 else np.sum(masked_data) / np.sum(mask)
                )
        final_tsv = pd.DataFrame(
            {
                "index": range(len(average_region)),
                "label_name": annotation.get_lateralized_region_names(left_first=True),
                "mean_scalar": average_region,
            }
        )
        filename_atlas_tsv = (output_dir or Path.cwd()) / f"{atlas_name}.tsv"
        filename_tsv.append(filename_atlas_tsv)
        final_tsv.to_csv(
            filename_atlas_tsv,
            sep="\t",
            index=False,
            columns=["index", "label_name", "mean_scalar"],
        )
    return filename_tsv
