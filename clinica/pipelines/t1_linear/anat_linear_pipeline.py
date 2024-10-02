# Use hash instead of parameters for iterables folder names
# Otherwise path will be too long and generate OSError
from pathlib import Path
from typing import List

from nipype import config

from clinica.pipelines.engine import Pipeline

cfg = dict(execution={"parameterize_dirs": False})
config.update_config(cfg)


class AnatLinear(Pipeline):
    """Anat Linear - Affine registration of anat (t1w or flair) images to standard space.

    This preprocessing pipeline includes globally three steps:
    1) Bias correction with N4 algorithm from ANTs.
    2) Linear registration to MNI152NLin2009cSym template with
       RegistrationSynQuick from ANTs.
    3) Crop the background (in order to save computational power).

    Returns:
        A clinica pipeline object containing the  AnatLinear pipeline.
    """

    @staticmethod
    def get_processed_images(
        caps_directory: Path, subjects: List[str], sessions: List[str]
    ) -> List[str]:
        from clinica.utils.filemanip import extract_image_ids
        from clinica.utils.input_files import T1W_LINEAR_CROPPED
        from clinica.utils.inputs import clinica_file_reader

        image_ids: List[str] = []
        if caps_directory.is_dir():
            cropped_files, _ = clinica_file_reader(
                subjects,
                sessions,
                caps_directory,
                T1W_LINEAR_CROPPED,
            )
            image_ids = extract_image_ids(cropped_files)
        return image_ids

    def _check_custom_dependencies(self) -> None:
        """Check dependencies that can not be listed in the `info.json` file."""
        pass

    def _check_pipeline_parameters(self) -> None:
        """Check pipeline parameters."""
        pass

    def get_input_fields(self) -> List[str]:
        """Specify the list of possible inputs of this pipeline.

        Returns
        -------
        list of str :
            A list of (string) input fields name.
        """
        return ["anat"]

    def get_output_fields(self) -> List[str]:
        """Specify the list of possible outputs of this pipeline.

        Returns
        -------
        list of str:
            A list of (string) output fields name.
        """
        return ["image_id"]

    def _build_input_node(self):
        """Build and connect an input node to the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe

        from clinica.utils.exceptions import ClinicaBIDSError, ClinicaException
        from clinica.utils.filemanip import extract_subjects_sessions_from_filename
        from clinica.utils.image import get_mni_template
        from clinica.utils.input_files import T1W_NII, Flair_T2W_NII
        from clinica.utils.inputs import clinica_file_filter
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_images_to_process

        self.ref_template = get_mni_template(
            "t1" if self.name == "t1-linear" else "flair"
        )

        # Display image(s) already present in CAPS folder
        # ===============================================
        processed_ids = self.get_processed_images(
            self.caps_directory, self.subjects, self.sessions
        )
        if len(processed_ids) > 0:
            cprint(
                msg=f"Clinica found {len(processed_ids)} image(s) already processed in CAPS directory:",
                lvl="warning",
            )
            for image_id in processed_ids:
                cprint(msg=f"{image_id.replace('_', ' | ')}", lvl="warning")
            cprint(msg=f"Image(s) will be ignored by Clinica.", lvl="warning")
            input_ids = [
                f"{p_id}_{s_id}" for p_id, s_id in zip(self.subjects, self.sessions)
            ]
            to_process_ids = list(set(input_ids) - set(processed_ids))
            self.subjects, self.sessions = extract_subjects_sessions_from_filename(
                to_process_ids
            )

        # Inputs from anat/ folder
        # ========================
        # anat image file:
        query = T1W_NII if self.name == "t1-linear" else Flair_T2W_NII

        anat_files, filtered_subjects, filtered_sessions = clinica_file_filter(
            self.subjects, self.sessions, self.bids_directory, query
        )
        self.subjects = filtered_subjects
        self.sessions = filtered_sessions

        if len(self.subjects):
            print_images_to_process(self.subjects, self.sessions)
            cprint("The pipeline will last approximately 6 minutes per image.")

        read_node = npe.Node(
            name="ReadingFiles",
            iterables=[
                ("anat", anat_files),
            ],
            synchronize=True,
            interface=nutil.IdentityInterface(fields=self.get_input_fields()),
        )
        self.connect(
            [
                (read_node, self.input_node, [("anat", "anat")]),
            ]
        )

    def _build_output_node(self):
        """Build and connect an output node to the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from nipype.interfaces.io import DataSink

        from clinica.utils.nipype import container_from_filename, fix_join

        if self.name == "flair-linear":
            from .anat_linear_utils import (
                get_substitutions_datasink_flair as get_substitutions,
            )
        else:
            from .anat_linear_utils import (
                get_substitutions_datasink_t1_linear as get_substitutions,
            )

        # Writing node
        write_node = npe.Node(name="WriteCaps", interface=DataSink())
        write_node.inputs.base_directory = str(self.caps_directory)
        write_node.inputs.parameterization = False

        # Other nodes
        # =====================================
        # Get substitutions to rename files
        get_ids = npe.Node(
            interface=nutil.Function(
                input_names=["bids_image_id"],
                output_names=["substitutions"],
                function=get_substitutions,
            ),
            name="GetIDs",
        )
        # Find container path from t1w filename
        container_path = npe.Node(
            nutil.Function(
                input_names=["bids_or_caps_filename"],
                output_names=["container"],
                function=container_from_filename,
            ),
            name="ContainerPath",
        )
        self.connect(
            [
                (self.input_node, container_path, [("anat", "bids_or_caps_filename")]),
                (self.output_node, get_ids, [("image_id", "bids_image_id")]),
                (
                    container_path,
                    write_node,
                    [
                        (
                            ("container", fix_join, self.name.replace("-", "_")),
                            "container",
                        )
                    ],
                ),
                (get_ids, write_node, [("substitutions", "substitutions")]),
                (self.output_node, write_node, [("image_id", "@image_id")]),
                (self.output_node, write_node, [("outfile_reg", "@outfile_reg")]),
            ]
        )

        if not (self.parameters.get("uncropped_image")):
            self.connect(
                [
                    (self.output_node, write_node, [("outfile_crop", "@outfile_crop")]),
                ]
            )

    def _build_core_nodes(self):
        """Build and connect the core nodes of the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        from nipype.interfaces import ants

        from clinica.pipelines.tasks import crop_nifti_task, get_filename_no_ext_task

        from .anat_linear_utils import print_end_pipeline

        image_id_node = npe.Node(
            interface=nutil.Function(
                input_names=["filename"],
                output_names=["image_id"],
                function=get_filename_no_ext_task,
            ),
            name="ImageID",
        )

        # The core (processing) nodes
        # =====================================

        # 1. N4biascorrection by ANTS. It uses nipype interface.
        n4biascorrection = npe.Node(
            name="n4biascorrection",
            interface=ants.N4BiasFieldCorrection(dimension=3, save_bias=True),
        )

        if self.name == "t1-linear":
            n4biascorrection.inputs.bspline_fitting_distance = 600
        else:
            n4biascorrection.inputs.bspline_fitting_distance = 100

        # 2. Modified registration - synquick reproduction
        ants_registration_node = npe.Node(
            name="antsRegistration", interface=ants.Registration()
        )
        ants_registration_node.inputs.fixed_image = self.ref_template
        ants_registration_node.inputs.interpolation = "Linear"
        ants_registration_node.inputs.dimension = 3
        ants_registration_node.inputs.winsorize_lower_quantile = 0.005
        ants_registration_node.inputs.winsorize_upper_quantile = 0.995
        ants_registration_node.inputs.transforms = ["Rigid", "Affine"]
        ants_registration_node.inputs.transform_parameters = [
            (0.1,),
            (0.1,),
        ]
        ants_registration_node.inputs.metric = ["MI"] * 2
        ants_registration_node.inputs.metric_weight = [1] * 2
        ants_registration_node.inputs.radius_or_number_of_bins = [32] * 2
        ants_registration_node.inputs.sampling_strategy = ["Regular", "Regular"]
        ants_registration_node.inputs.sampling_percentage = [0.25, 0.25]
        ants_registration_node.inputs.number_of_iterations = [
            [40, 0, 0, 0],
            [20, 0, 0, 0],
        ]
        ants_registration_node.inputs.convergence_threshold = [1.0e-6] * 2
        ants_registration_node.inputs.convergence_window_size = [10] * 2
        ants_registration_node.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 2
        ants_registration_node.inputs.sigma_units = ["vox"] * 2
        ants_registration_node.inputs.shrink_factors = [[8, 4, 2, 1]] * 2
        ants_registration_node.inputs.use_histogram_matching = False
        ants_registration_node.inputs.output_warped_image = True

        if random_seed := self.parameters.get("random_seed", None):
            ants_registration_node.inputs.random_seed = random_seed

        # 3. Crop image (using nifti). It uses custom interface, from utils file

        cropnifti = npe.Node(
            name="cropnifti",
            interface=nutil.Function(
                function=crop_nifti_task,
                input_names=["input_image", "output_path"],
                output_names=["output_img"],
            ),
        )
        cropnifti.inputs.output_path = self.base_dir

        # 4. Print end message
        print_end_message = npe.Node(
            interface=nutil.Function(
                input_names=["anat", "final_file"], function=print_end_pipeline
            ),
            name="WriteEndMessage",
        )
        self.connect(
            [
                (self.input_node, image_id_node, [("anat", "filename")]),
                (self.input_node, n4biascorrection, [("anat", "input_image")]),
                (
                    n4biascorrection,
                    ants_registration_node,
                    [("output_image", "moving_image")],
                ),
                (
                    image_id_node,
                    ants_registration_node,
                    [("image_id", "output_transform_prefix")],
                ),
                # Connect to DataSink
                (image_id_node, self.output_node, [("image_id", "image_id")]),
                (
                    ants_registration_node,
                    self.output_node,
                    [("warped_image", "outfile_reg")],
                ),
                (self.input_node, print_end_message, [("anat", "anat")]),
            ]
        )
        if not (self.parameters.get("uncropped_image")):
            self.connect(
                [
                    (
                        ants_registration_node,
                        cropnifti,
                        [("warped_image", "input_image")],
                    ),
                    (cropnifti, self.output_node, [("output_img", "outfile_crop")]),
                    (cropnifti, print_end_message, [("output_img", "final_file")]),
                ]
            )
        else:
            self.connect(
                [
                    (
                        ants_registration_node,
                        print_end_message,
                        [("warped_image", "final_file")],
                    ),
                ]
            )
