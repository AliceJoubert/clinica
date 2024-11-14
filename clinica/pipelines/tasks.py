"""This module contains Nipype tasks used in several pipelines."""


def crop_nifti_task(input_image: str, output_path: str) -> str:
    from pathlib import Path

    from clinica.utils.image import crop_nifti

    return str(crop_nifti(Path(input_image), Path(output_path)))


def get_filename_no_ext_task(filename: str) -> str:
    from pathlib import Path

    from clinica.utils.filemanip import get_filename_no_ext

    return get_filename_no_ext(Path(filename))


def get_rigid(filename: str) -> str:
    import re
    from pathlib import Path
    # ex : sub-ADNI002S0413_ses-M12_T1w.nii.gz

    CAPSr = Path("/Users/alice.joubert/clinicaQC/data/BE_method_evaluation/ADNI/CAPS_r")
    pattern = f"{Path(filename).name.replace('_T1', '')}.*.mat"

    possibilities = CAPSr.rglob("*.mat")
    mat = [p for p in possibilities if re.search(pattern, p)]

    if len(mat) == 1:
        mat = mat[0]
    else:
        raise ValueError(f"Expected only one matrix, got : {mat}")

    # todo :connect
    return str(mat)
