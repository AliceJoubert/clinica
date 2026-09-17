from clinica.pipelines.utils import FreeSurferAnnotationImage
from clinica.utils.image import HemiSphere


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
