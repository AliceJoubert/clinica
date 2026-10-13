ARG CLINICA_VERSION=0.11.0
ARG PYTHON_VERSION=3.11
ARG SPM_VERSION=25.01.02
ARG MCR_VERSION=R2024b
ARG FREESURFER_VERSION=7.4.1
ARG PETPVC_VERSION=1.2.12

# Stage 0 : Freesurfer
FROM ubuntu:jammy-20240808 AS freesurfer
ARG FREESURFER_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends bzip2 ca-certificates unzip curl
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FREESURFER_VERSION}/freesurfer-linux-ubuntu22_amd64-${FREESURFER_VERSION}.tar.gz \
     | tar zxv --no-same-owner -C /opt

# Stage 1 : SPM
FROM ghcr.io/spm/spm-docker:docker-matlab-${SPM_VERSION} AS spm-builder

# Stage 2 : clinica
FROM python:${PYTHON_VERSION}-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials
RUN apt-get update && apt-get install -y tcsh perl bc dc gawk vim git build-essential xorg cmake wget

# Define working directory
RUN useradd -ms /bin/bash clinica_user
USER clinica_user
ENV WORKDIR=/opt/app_clinica
WORKDIR ${WORKDIR}

# Copy from previous stage
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer
ENV FREESURFER_HOME="/opt/freesurfer"
ENV OS="Linux"
ENV FS_OVERRIDE=0
ENV FIX_VERTEX_AREA=""
ENV FSF_OUTPUT_FORMAT="nii.gz"
ENV SUBJECTS_DIR="${FREESURFER_HOME}/subjects"
ENV FUNCTIONALS_DIR="${FREESURFER_HOME}/sessions"
ENV MNI_DIR="${FREESURFER_HOME}/mni"
ENV LOCAL_DIR="${FREESURFER_HOME}/local"
ENV MINC_BIN_DIR="${FREESURFER_HOME}/mni/bin"
ENV MINC_LIB_DIR="${FREESURFER_HOME}/mni/lib"
ENV MNI_DATAPATH="${FREESURFER_HOME}/mni/data"
ENV PERL5LIB="${FREESURFER_HOME}/mni/lib/perl5"
ENV MNI_PERL5LIB="${FREESURFER_HOME}/mni/lib/perl5"

COPY --from=spm-builder /opt/spm /opt/spm
COPY --from=spm-builder /usr/local/MATLAB/MATLAB_Runtime/${MCR_VERSION} /opt/mcr
ENV PATH="${FREESURFER_HOME}/bin:/opt/spm:${PATH}"
ENV SPMSTANDALONE_HOME="/opt/spm"
ENV MCR_HOME="/opt/mcr"
ENV FORCE_SPMPMCR=1
ENV SPM_HTML_BROWSER=0
ENV MCR_CACHE_ROOT="/tmp/mcr_cache"
ENV LD_LIBRARY_PATH="${MCR_HOME}/extern/bin/glnxa64:${MCR_HOME}/bin/glnxa64:${MCR_HOME}/sys/os/glnxa64:${MCR_HOME}/runtime/glnxa64:${LD_LIBRARY_PATH}"

# Install miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O ${WORKDIR}/miniforge.sh
RUN mkdir -p ${WORKDIR}/.conda
RUN bash ${WORKDIR}/miniforge.sh -b -p "${WORKDIR}/miniforge"
ENV PATH="${WORKDIR}/miniforge/bin:${PATH}"

# Create environment
ARG CLINICA_VERSION
ARG PYTHON_VERSION
ARG PETPVC_VERSION
RUN conda create -y -k -n clinica-env python=${PYTHON_VERSION} petpvc=${PETPVC_VERSION} -c conda-forge
RUN conda run -n clinica-env pip install clinica==${CLINICA_VERSION}
RUN echo "source activate clinica-env" >> ~/.bashrc
ENV PATH="${WORKDIR}/miniforge/envs/clinica-env/bin:${PATH}"

RUN chmod -R 777 ${WORKDIR}

# Use clinica CLI as default
ENTRYPOINT ["clinica", "run"]
CMD ["--help"]
