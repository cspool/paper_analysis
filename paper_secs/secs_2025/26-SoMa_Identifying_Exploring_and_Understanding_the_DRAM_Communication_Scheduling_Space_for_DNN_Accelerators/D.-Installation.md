# *D. Installation*

For artifact evaluation, start by downloading the artifact from Zenodo:

```
$ wget -O SOMA_AE.zip https://zenodo.org/records
    /14599935/files/SOMA_AE.zip?download=1
$ unzip SOMA_AE.zip
```

Our SoMa exploration framework is in "SOMA". We use 'build.sh' to build the SoMa framework and create the result directory.

```
$ cd SOMA
$ ./build.sh
```

The executable target will be generated at "./build/soma", and the result directories will be "results/overall" and "results/dse".

You can install the needed Python packages using pip with the following commands.

```
$ pip install -r requirements.txt
```

Or you can use conda to install with the following commands.

```
$ conda install --file requirements.txt
```

