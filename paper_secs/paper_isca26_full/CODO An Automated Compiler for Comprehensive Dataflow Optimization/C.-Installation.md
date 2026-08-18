# *C. Installation*

The docker image can be downloaded from the Docker Hub repository by the instructions below:

```
$ docker pull xzz11/codo_ae_image:v1
```

When building a new docker container, the directory of Vitis HLS and Vivado need to be mounted. You can verify your Vitis directory with the following instructions:

```
$ ls $(YOUR_VITIS_DIR)
 DocNav Downloads Model_Composer
 Vitis Vitis_HLS Vivado xic
```

Then the docker container can be built from the provided docker image:

```
$ docker run -it -v $(YOUR_VITIS_DIR):$
 (YOUR_VITIS_DIR) -e LC_ALL=en_US.UTF-8
 -e LANG=en_US.UTF-8
 xzz11/codo_ae_image:v1 /bin/bash
```

# *C. Installation*

The docker image can be downloaded from the Docker Hub repository by the instructions below:

```
$ docker pull xzz11/codo_ae_image:v1
```

When building a new docker container, the directory of Vitis HLS and Vivado need to be mounted. You can verify your Vitis directory with the following instructions:

```
$ ls $(YOUR_VITIS_DIR)
 DocNav Downloads Model_Composer
 Vitis Vitis_HLS Vivado xic
```

Then the docker container can be built from the provided docker image:

```
$ docker run -it -v $(YOUR_VITIS_DIR):$
 (YOUR_VITIS_DIR) -e LC_ALL=en_US.UTF-8
 -e LANG=en_US.UTF-8
 xzz11/codo_ae_image:v1 /bin/bash
```

