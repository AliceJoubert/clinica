"""Convert OASIS2 dataset (...) to BIDS."""
# todo : change link
from typing import Optional

from clinica.utils.filemanip import UserProvidedPath

__all__ = ["convert"]


def convert(
    path_to_dataset: UserProvidedPath,
    bids_dir: UserProvidedPath,
    path_to_clinical: UserProvidedPath,
    subjects: Optional[UserProvidedPath] = None,
    n_procs: Optional[int] = 1,
    **kwargs,
):
    """Convert the entire dataset in BIDS.

    Scans available files in the path_to_dataset,
    identifies the patients that have images described by the JSON file,
    converts the image with the highest quality for each category.
    """
    from clinica.utils.stream import cprint

    from .._utils import validate_input_path, write_modality_agnostic_files
    from ..factory import get_converter_name
    from ..study_models import StudyName

    path_to_dataset = validate_input_path(path_to_dataset)
    bids_dir = validate_input_path(bids_dir, check_exist=False)
    path_to_clinical = validate_input_path(path_to_clinical)

    if subjects:
        cprint(
            (
                f"Subject filtering is not yet implemented in {get_converter_name(StudyName.OASIS2)} converter. "
                "All subjects available will be converted."
            ),
            lvl="warning",
        )
    if n_procs != 1:
        cprint(
            f"{get_converter_name(StudyName.NIFD)} converter does not support multiprocessing yet. n_procs set to 1.",
            lvl="warning",
        )

    # todo : read clinical data
    # todo : read imaging data
    # todo : write in the bids folder

    # todo : change readme data
    readme_data = {
        "link": "",
        "desc": (""),
    }
    write_modality_agnostic_files(
        study_name=StudyName.OASIS2,
        readme_data=readme_data,
        bids_dir=bids_dir,
    )
    cprint("Conversion to BIDS succeeded.", lvl="info")


# todo : mention Nikhil
