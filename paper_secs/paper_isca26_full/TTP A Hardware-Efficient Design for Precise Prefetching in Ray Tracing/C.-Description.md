# *C. Description*

- *1) How to access:* We provide the Docker image which has everything required to run the simulations and generate the figures. You can download it from Zenodo using the archived link.
- *2) Hardware dependencies:* Only requirement is 32+GB of memory.
- *3) Software dependencies:* Only a Docker installation is required. All software dependencies are installed in the Docker image.

#### *D. Installation*

Download the docker image from Zenodo, and start a container using the commands below,

\$ docker load < ttp-isca2026-ae.tar.gz \$ docker run -it ttp-isca2026-ae:1.0 /bin/bash

