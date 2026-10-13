variable "PYTHON_VERSION" {
    default = "3.11"
    validation {
        condition = contains(["3.10", "3.11", "3.12", "3.13", "3.14"], PYTHON_VERSION)
  }
}

variable "CLINICA_VERSION" {
    default = "0.12.0"
    validation {
        condition = contains(["0.12.0", "0.11.3"],CLINICA_VERSION)
  }
}

variable "ANTS_VERSION" {
    default = "2.6.5"
}

variable "SPM_VERSION" { default = "25.01.02" }
variable "MCR_VERSION" {
    default = "R2024b"
    validation = MCR_VERSION == "R2024b"
}
variable "FREESURFER_VERSION" { default = "7.4.1"}
variable "PETPVC_VERSION" { default = "1.2.12"}
variable "FSL_VERSION" { default = ""}
variable "MRTRIX_VERSION" { default = ""}
variable "CONVERT3D_VERSION" { default = ""}
