:todo: in installation talk about this page
:warning: only built in amd and only for clinica >=0.11.0
:todo: here : 
- users
- how to call
- advanced usage : change the version with bake, how to
- link to dockerhub
- auto publish on dockerhub

maybe : 
- enrich Makefile to build
- GHCR vs dockerhub ? GHCR free for public images
- Freesurfer ; Check if their terms permit you to redistribute the binaries at all
--> If redistribution of something is dubious, ship a Dockerfile that downloads it at build time and let users build locally — that's a materially different legal position from you hosting the binaries.
- Apptainer/Singularity
- PR with tags for CI triggers
- flags for docker build again vs cache

Each containers contains the third-party software required to run the pipelines :

=== "core"
    - all converters
    - t1-linear
    - pet-linear

=== "volume"
    - t1-volume
    - pet-volume
    - stats-volume

=== "surface"
    - t1-freesurfer
    - pet-surface
    - stats-surface

=== "dwi (coming later)"
    all dwi pipelines

Time to build on mac :

=== "core"
    5 minuted

=== "volume"
    s

=== "surface"
    s

=== "dwi (coming later)"
    ?


Default versions. 
How to use with clinica command lines ? 
How to mount volumes ? 
How to give the freesurfer license ?
How to build locally to change the version with bake.
How to run with a user
Rootless docker possible ? 
Difference between rootless docker
Transition to singularity ?

:time: Coming later :
- clinica-docker-cli
- dwi pipelines



bake --print to see the configuration
default in bake is core. 
docker compose for the volumes definition ?
possible to bake without cloning (no make available)

todo : in it-guide release section check the bake

