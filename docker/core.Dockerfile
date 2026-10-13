ARG ANTS_VERSION=2.6.5
ARG PYTHON_VERSION=3.11
ARG CLINICA_VERSION=0.11.0

# ---- Stage 1: grab ANTs binaries ----
FROM antsx/ants:$ANTS_VERSION AS ants-builder

# ---- Stage 2: your actual image ----
FROM python:${PYTHON_VERSION}-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials (add shared libs ANTs needs at runtime)
RUN apt-get update && apt-get install -y \
    git build-essential cmake wget ca-certificates \
    zlib1g-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy ANTs from stage 1
COPY --from=ants-builder /opt/ants /opt/ants

ENV PATH="/opt/ants/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/ants/lib:${LD_LIBRARY_PATH}"

# Define working directory
RUN useradd -ms /bin/bash clinica_user
USER clinica_user
ENV WORKDIR=/opt/app_clinica
WORKDIR ${WORKDIR}

# Install miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O ${WORKDIR}/miniforge.sh
RUN mkdir -p ${WORKDIR}/.conda
RUN bash ${WORKDIR}/miniforge.sh -b -p "${WORKDIR}/miniforge"
ENV PATH="${WORKDIR}/miniforge/bin:${PATH}"

# Create environment
ARG PYTHON_VERSION
ARG CLINICA_VERSION
RUN conda create -y -k -n clinica-env python=$PYTHON_VERSION -c conda-forge
RUN conda run -n clinica-env pip install clinica==$CLINICA_VERSION
RUN echo "source activate clinica-env" > ~/.bashrc
ENV PATH="${WORKDIR}/miniforge/envs/clinica-env/bin:${PATH}"

RUN chmod -R 777 ${WORKDIR}

# Use clinica CLI as default
ENTRYPOINT ["clinica", "run"]
CMD ["--help"]

