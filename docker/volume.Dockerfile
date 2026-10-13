ARG CLINICA_VERSION=0.11.0
ARG PYTHON_VERSION=3.11
ARG SPM_VERSION=25.01.02
ARG MCR_VERSION=R2024b

# Stage 1 : SPM
FROM ghcr.io/spm/spm-docker:docker-matlab-${SPM_VERSION} AS spm-builder

# Stage 2 : clinica
FROM python:${PYTHON_VERSION}-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials
RUN apt-get update && apt-get install -y git build-essential xorg cmake wget ca-certificates

# Define working directory
RUN useradd -ms /bin/bash clinica_user
USER clinica_user
ENV WORKDIR=/opt/app_clinica
WORKDIR ${WORKDIR}

# Copy from previous stage
ARG MCR_VERSION
COPY --from=spm-builder /opt/spm /opt/spm
COPY --from=spm-builder /usr/local/MATLAB/MATLAB_Runtime/${MCR_VERSION} /opt/mcr
ENV PATH="/opt/spm:${PATH}"
ENV SPMSTANDALONE_HOME="/opt/spm"
ENV MCR_HOME="/opt/mcr"
ENV FORCE_SPMPMCR=1
ENV SPM_HTML_BROWSER=0
ENV LD_LIBRARY_PATH="${MCR_HOME}/extern/bin/glnxa64:${MCR_HOME}/bin/glnxa64:${MCR_HOME}/sys/os/glnxa64:${MCR_HOME}/runtime/glnxa64:${LD_LIBRARY_PATH}"

# Install miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O ${WORKDIR}/miniforge.sh
RUN mkdir -p ${WORKDIR}/.conda
RUN bash ${WORKDIR}/miniforge.sh -b -p "${WORKDIR}/miniforge"
ENV PATH="${WORKDIR}/miniforge/bin:${PATH}"

# Create environment
ARG PYTHON_VERSION
ARG CLINICA_VERSION
RUN conda create -y -k -n clinica-env python=${PYTHON_VERSION} -c conda-forge
RUN conda run -n clinica-env pip install clinica==${CLINICA_VERSION}
RUN echo "source activate clinica-env" >> ~/.bashrc
ENV PATH="${WORKDIR}/miniforge/envs/clinica-env/bin:${PATH}"

RUN chmod -R 777 ${WORKDIR}

# Use clinica CLI as default
ENTRYPOINT ["clinica", "run"]
CMD ["--help"]
