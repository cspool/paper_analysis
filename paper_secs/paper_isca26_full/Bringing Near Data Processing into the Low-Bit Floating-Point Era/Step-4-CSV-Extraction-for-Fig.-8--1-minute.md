# Step 4: CSV Extraction for Fig. 8 (< 1 minute)

<sup>1</sup> python ./scripts/final/3\_single\_op\_with\_predictor/ extract\_fig8.py

Extracted CSV files for different sub-figures in Fig. 8 are saved to 3\_single\_op\_with\_predictor/fig8/. Latencies are reported in cycles in the CSV files. To convert to nanoseconds, multiply by tCK found in flexq\_ndp/config.

#### *2) Energy Consumption (Table VI):* takes ∼1 hour.

Energy consumption is estimated from simulation traces and power parameters following the equations in DRAMSim3 [38], and is reported in pJ (picojoule) in the output file.

```
1 # run energy simulation in the background
2 bash ./scripts/final/8_energy/bash_energy.sh
3 # extract results
4 python ./scripts/final/8_energy/extract_energy.py
```

Results are written to 8\_energy/energy\_mvm.csv.

## *3) Comparison between precisions and QConfigs (Fig. 9) :* takes <1 minute.

```
1 python ./scripts/final/3_single_op_with_predictor/
     extract_all.py # extract all result
2 python ./scripts/final/3_single_op_with_predictor/
     extract_fig9.py # extract data for fig.9
```

Results: 3\_single\_op\_with\_predictor/fig9/.

#### *4) CNN Operators (Table VIII):* takes ∼1 hour.

```
1 bash ./scripts/final/3_single_op_with_predictor/cnn.
     sh # 128 processes
```

Results: 3\_single\_op\_with\_predictor/log\_cnn/.

*5) End-to-End Latency (Fig. 10):* takes ∼0.5 hour. Arranged in four sequential steps, and each step must wait until the preceding step, including any background processes, has fully completed.

```
1 # Step 1: extract single-operator data (requires
      GEMM speedup experiment (Fig.8))
2 python ./scripts/final/3_single_op_with_predictor/
      extract_all.py
4 # Step 2: run FP16 baseline in the background
5 bash ./scripts/fp16_baseline/process_workload_all.sh
6 ps -aux | grep -c for_quant # wait until all
      processes finish
8 # Step 3: get online quantisation latency in the
9 bash ./scripts/final/3_single_op_with_predictor/
      quant_latency/quant_lat.sh
10 ps -aux | grep -c test_quant_latency # wait until
      all processes finish
12 # Step 4: compute end-to-end latency and plot
13 python ./scripts/final/4_e2e/calculate_e2e_lat.py
```

The end-to-end results (cycles and speedup) are printed directly to the terminal. Sub-figures for Fig. 10 are printed to 4\_e2e/latency\_plots/.

## *6) Ablation Study (Table IX):* takes ∼5 hours

```
1 bash ./scripts/final/7_ablation/ablation.sh # 128
     processes
2 # extract results
3 python ./scripts/final/7_ablation/extract_ablation.
     py
```

Results: 7\_ablation/ablation\_summary.csv.

#### *7) Buffer Size Sensitivity (Fig. 11):* takes ∼8 hours.

```
1 bash ./scripts/final/5_buffer_change/buffer_change.
     sh # 128 processes
2 # extract results
3 python ./scripts/final/5_buffer_change/
     extract_buffer_change.py
```

Results: 5\_buffer\_change/fig11a.csv and 5\_buffer\_change/fig11b.csv.

#### *8) FP32 Throughput Sensitivity (Fig. 12):* takes ∼1 day.

```
1 bash ./scripts/final/6_throughput_change/
     throughput_change.sh # 128 processes
2 # extract results
3 python ./scripts/final/6_throughput_change/
     extract_throughput_speedup.py
```

Results: 6\_throughput\_change/ throughput\_speedup.csv

## *9) Mixed-Precision PU Compatibility (Fig. 13):* takes ∼0.5 hours.

```
1 bash ./scripts/final/9_mix_precision_pu/mix_pre_pu.
     sh # 128 processes
2 # extract results
3 python ./scripts/final/9_mix_precision_pu/
     extract_mix_precision_pu.py
```

Results: 9\_mix\_precision\_pu/ mix\_precision\_pu\_summary.csv.

#### *F. Evaluation and expected results*

The results generated/plotted from this artifact should match those shown in result figures and tables.

#### *G. Experiment customization*

We already provide experiments for different hardware parameters (buffer sizes, throughput, different kind of PU). Users can extend on these experiments to explore more hardware variants, or explore different combinations of hardware and quantization configurations.

For example, for mixed-precision PU experiment, user can modify the configuration file ( flexq\_ndp/scripts/ final/9\_mix\_precision\_pu/workload/ quant.yaml) to test on more quantization configurations.

# Step 4: CSV Extraction for Fig. 8 (< 1 minute)

<sup>1</sup> python ./scripts/final/3\_single\_op\_with\_predictor/ extract\_fig8.py

Extracted CSV files for different sub-figures in Fig. 8 are saved to 3\_single\_op\_with\_predictor/fig8/. Latencies are reported in cycles in the CSV files. To convert to nanoseconds, multiply by tCK found in flexq\_ndp/config.

#### *2) Energy Consumption (Table VI):* takes ∼1 hour.

Energy consumption is estimated from simulation traces and power parameters following the equations in DRAMSim3 [38], and is reported in pJ (picojoule) in the output file.

```
1 # run energy simulation in the background
2 bash ./scripts/final/8_energy/bash_energy.sh
3 # extract results
4 python ./scripts/final/8_energy/extract_energy.py
```

Results are written to 8\_energy/energy\_mvm.csv.

## *3) Comparison between precisions and QConfigs (Fig. 9) :* takes <1 minute.

```
1 python ./scripts/final/3_single_op_with_predictor/
     extract_all.py # extract all result
2 python ./scripts/final/3_single_op_with_predictor/
     extract_fig9.py # extract data for fig.9
```

Results: 3\_single\_op\_with\_predictor/fig9/.

#### *4) CNN Operators (Table VIII):* takes ∼1 hour.

```
1 bash ./scripts/final/3_single_op_with_predictor/cnn.
     sh # 128 processes
```

Results: 3\_single\_op\_with\_predictor/log\_cnn/.

*5) End-to-End Latency (Fig. 10):* takes ∼0.5 hour. Arranged in four sequential steps, and each step must wait until the preceding step, including any background processes, has fully completed.

```
1 # Step 1: extract single-operator data (requires
      GEMM speedup experiment (Fig.8))
2 python ./scripts/final/3_single_op_with_predictor/
      extract_all.py
4 # Step 2: run FP16 baseline in the background
5 bash ./scripts/fp16_baseline/process_workload_all.sh
6 ps -aux | grep -c for_quant # wait until all
      processes finish
8 # Step 3: get online quantisation latency in the
9 bash ./scripts/final/3_single_op_with_predictor/
      quant_latency/quant_lat.sh
10 ps -aux | grep -c test_quant_latency # wait until
      all processes finish
12 # Step 4: compute end-to-end latency and plot
13 python ./scripts/final/4_e2e/calculate_e2e_lat.py
```

The end-to-end results (cycles and speedup) are printed directly to the terminal. Sub-figures for Fig. 10 are printed to 4\_e2e/latency\_plots/.

## *6) Ablation Study (Table IX):* takes ∼5 hours

```
1 bash ./scripts/final/7_ablation/ablation.sh # 128
     processes
2 # extract results
3 python ./scripts/final/7_ablation/extract_ablation.
     py
```

Results: 7\_ablation/ablation\_summary.csv.

#### *7) Buffer Size Sensitivity (Fig. 11):* takes ∼8 hours.

```
1 bash ./scripts/final/5_buffer_change/buffer_change.
     sh # 128 processes
2 # extract results
3 python ./scripts/final/5_buffer_change/
     extract_buffer_change.py
```

Results: 5\_buffer\_change/fig11a.csv and 5\_buffer\_change/fig11b.csv.

#### *8) FP32 Throughput Sensitivity (Fig. 12):* takes ∼1 day.

```
1 bash ./scripts/final/6_throughput_change/
     throughput_change.sh # 128 processes
2 # extract results
3 python ./scripts/final/6_throughput_change/
     extract_throughput_speedup.py
```

Results: 6\_throughput\_change/ throughput\_speedup.csv

## *9) Mixed-Precision PU Compatibility (Fig. 13):* takes ∼0.5 hours.

```
1 bash ./scripts/final/9_mix_precision_pu/mix_pre_pu.
     sh # 128 processes
2 # extract results
3 python ./scripts/final/9_mix_precision_pu/
     extract_mix_precision_pu.py
```

Results: 9\_mix\_precision\_pu/ mix\_precision\_pu\_summary.csv.

#### *F. Evaluation and expected results*

The results generated/plotted from this artifact should match those shown in result figures and tables.

#### *G. Experiment customization*

We already provide experiments for different hardware parameters (buffer sizes, throughput, different kind of PU). Users can extend on these experiments to explore more hardware variants, or explore different combinations of hardware and quantization configurations.

For example, for mixed-precision PU experiment, user can modify the configuration file ( flexq\_ndp/scripts/ final/9\_mix\_precision\_pu/workload/ quant.yaml) to test on more quantization configurations.

