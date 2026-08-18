# *E. Experiment workflow*

The artifact runs as a single Docker container that executes the full benchmark sweep:

```
$ mkdir results
$ docker run --rm -v $(pwd)/results:/output \
    icepack-artifact
```

The pipeline generates syndrome data, applies the three compression stages, and produces 4 CSV files and 4 PNG figures in the results/ directory corresponding to Figures 5, 7, 8, and 15 in the paper.

A smoke test mode is available to verify the setup before the full run:

```
$ docker run --rm -e SMOKE_TEST=1 \
    -v $(pwd)/results:/output icepack-artifact
```

# *E. Experiment workflow*

The artifact runs as a single Docker container that executes the full benchmark sweep:

```
$ mkdir results
$ docker run --rm -v $(pwd)/results:/output \
    icepack-artifact
```

The pipeline generates syndrome data, applies the three compression stages, and produces 4 CSV files and 4 PNG figures in the results/ directory corresponding to Figures 5, 7, 8, and 15 in the paper.

A smoke test mode is available to verify the setup before the full run:

```
$ docker run --rm -e SMOKE_TEST=1 \
    -v $(pwd)/results:/output icepack-artifact
```

