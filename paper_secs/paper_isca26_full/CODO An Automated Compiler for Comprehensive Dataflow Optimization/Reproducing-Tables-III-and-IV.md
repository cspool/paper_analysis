# Reproducing Tables III and IV:

```
$ cd experiments/table-3\_and\_table-4/
$ bash run_ae.sh
$ bash run_hls.sh
$ python3 batch_extract_rpt_metrics.py
```

The synthesis time is approximately 512 minutes. Results are stored in result.csv.

On-board evaluation results (Table V, Table VI, and Figure 9):

The on-board experiments were not conducted for AE purposes, as generating all bitstreams for the evaluated models requires over two weeks and a properly configured U280 environment. Nevertheless, we provide the host and kernel source code, placement-and-route reports, and prebuilt xclbin files for GPT-2 (corresponding to Table VI and Figure 9).

```
$ cd <corresponding_folder>
$ ./host.exe kernel.hw.xclbin
```

# Reproducing Tables III and IV:

```
$ cd experiments/table-3\_and\_table-4/
$ bash run_ae.sh
$ bash run_hls.sh
$ python3 batch_extract_rpt_metrics.py
```

The synthesis time is approximately 512 minutes. Results are stored in result.csv.

On-board evaluation results (Table V, Table VI, and Figure 9):

The on-board experiments were not conducted for AE purposes, as generating all bitstreams for the evaluated models requires over two weeks and a properly configured U280 environment. Nevertheless, we provide the host and kernel source code, placement-and-route reports, and prebuilt xclbin files for GPT-2 (corresponding to Table VI and Figure 9).

```
$ cd <corresponding_folder>
$ ./host.exe kernel.hw.xclbin
```

