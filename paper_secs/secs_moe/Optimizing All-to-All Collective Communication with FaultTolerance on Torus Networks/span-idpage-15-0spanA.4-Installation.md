# <span id="page-15-0"></span>A.4 Installation

For experiments with analytical backend, we provide three scripts: the first to build the anaconda environment, the second to activate the environment, and the last to build ASTRA-SIM. The commands are shown below:

```
$ cd analytical_backend /
# Create anaconda environment and handle
    software dependencies
$ conda env create -f astra - sim - analytical .
    yml
```

```
# Enter the anaconda environment
$ conda activate astra - sim - analytical
# compile Astra - SIM with analytical backend
$ ./ build / astra_analytical / build . sh -c
```

For experiments with GARNET backend, we provide four scripts: the first to build the anaconda environment, the second to activate the environment, the third to configure protobuf, and the last to build Astra-SIM. The commands are shown below:

```
$ cd garnet_backend /
# Create anaconda environment and handle
    software dependencies
$ conda env create -f astra - sim - garnet . yml
# Enter the anaconda environment
$ conda activate astra - sim - garnet
# Protobuf configuration
$ bash setup_protobuf . sh
# compile Astra - SIM with GARNET backend
$ ./ build / astra_garnet / build . sh -c
```

