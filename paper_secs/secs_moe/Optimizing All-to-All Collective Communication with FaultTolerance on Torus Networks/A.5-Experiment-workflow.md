# A.5 Experiment workflow

We provide two click-to-run scripts to run all the simulation experiments with analytical backend and GARNET backend, respectively. The command is shown below:

```
# For analytical_backend
$ cd ./ analytical_backend / examples / scripts /
$ conda activate astra - sim - analytical
# Run all the simulations
$ bash run - all . sh
# For GARNET backend
$ cd ./ garnet_backend / examples / scripts /
$ conda activate astra - sim - garnet
# Run all the simulations
$ bash run - all . sh
```

We also provide a single script for clicking to run all the real machine experiments. The commands are shown below:

```
$ cd ./ real_machine /
# Run all the real machine tests
$ bash Run_All_to_All . sh
```

Finally, we provide the figure scripts. The commands are shown below:

```
# Draw the figures
$ conda activate astra - sim - analytical
$ bash plot - figure . sh
```

