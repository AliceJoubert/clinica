group "default" {
  targets = ["core"]
}

target "_common" {
  context   = "."
  platforms = ["linux/amd64"]

  args = {
    CLINICA_VERSION = CLINICA_VERSION
    PYTHON_VERSION = PYTHON_VERSION
  }

  labels = {
    "org.opencontainers.image.version"       = CLINICA_VERSION
    "org.opencontainers.image.source"        = "https://github.com/aramis-lab/clinica"
    "org.opencontainers.image.documentation" = "https://aramislab.paris.inria.fr/clinica/docs/public/latest/"
    "org.opencontainers.image.authors"       = "ARAMIS Lab - Paris Brain Institute"
    "org.opencontainers.image.licenses"      = "MIT"
  }
}

target "core" {
  inherits = ["_common"]
  dockerfile = "./docker/core.Dockerfile"
  tags = ["clinica-core:${CLINICA_VERSION}"]
  args = {
    ANTS_VERSION = ANTS_VERSION
  }
  labels = {
    "org.opencontainers.image.title" = "clinica-core"
  }
}

target "volume" {
  inherits = ["_common"]
  dockerfile = "./docker/volume.Dockerfile"
  tags = ["clinica-volume:CLINICA_VERSION"]
  args = {
    SPM_VERSION = SPM_VERSION
    MCR_VERSION = MCR_VERSION
  }
  labels = {
    "org.opencontainers.image.title" = "clinica-volume"
  }
}

target "surface" {
  inherits = ["_common"]
  dockerfile = "./docker/surface.Dockerfile"
  tags = ["clinica-surface:CLINICA_VERSION"]
  args = {
    SPM_VERSION = SPM_VERSION
    MCR_VERSION = MCR_VERSION
    FREESURFER_VERSION = FREESURFER_VERSION
    PETPVC_VERSION = PETPVC_VERSION
  }
  labels = {
    "org.opencontainers.image.title" = "clinica-surface"
  }
}

target "dwi" {
  inherits = ["_common"]
  dockerfile = "./docker/dwi.Dockerfile"
  tags = ["clinica-dwi:CLINICA_VERSION"]
  args = {
    FSL_VERSION = FSL_VERSION
    MRTRIX_VERSION = MRTRIX_VERSION
    FREESURFER_VERSION = FREESURFER_VERSION
    PETPVC_VERSION = PETPVC_VERSION
    ANTS_VERSION = ANTS_VERSION
    CONVERT3D_VERSION = CONVERT3D_VERSION
  }
  labels = {
    "org.opencontainers.image.title" = "clinica-dwi"
  }
}
