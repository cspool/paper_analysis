# *E. Experiment Workflow*

Note1: All command lines in Appendix XIII-E should be run from the flexq\_ndp project root (under flexq\_ndp directory). All result paths provided in the workflow is under flexq\_ndp/scripts/final directory.

Note2: Experiment for GEMM operator speedups (Fig. 8) must be completed before running any other experiments, as subsequent steps depend on its output.

*1) GEMM Operator Speedup (Fig. 8):* arranged in four steps, each step must wait for the preceding step is finished (including background processes).

Step 1: Search compilation strategies (∼1 day)

```
1 # run in the background
2 bash ./scripts/final/3_single_op_with_predictor/
     part_1.sh
3 # monitor processes running in the background
4 ps -aux | grep -c single_op_with_predictor_part1
```

Best strategies found during Part 1 are stored in 3\_single\_op\_with\_predictor/ tmp\_strategy/tmp\_strategy.yaml

Step 2: Simulate Baseline & FlexQ-NDP Latency (∼1 day)

```
1 bash ./scripts/final/3_single_op_with_predictor/
      part2_m1.sh # 94 configs
2 bash ./scripts/final/3_single_op_with_predictor/
      part2_m2.sh # 20 configs
3 bash ./scripts/final/3_single_op_with_predictor/
      part2_m4.sh # 86 configs
4 bash ./scripts/final/3_single_op_with_predictor/
      part2_m8.sh # 20 configs
5 bash ./scripts/final/3_single_op_with_predictor/
      part2_m16.sh # 94 configs
6 bash ./scripts/final/3_single_op_with_predictor/
      part2_m32.sh # 20 configs
7 bash ./scripts/final/3_single_op_with_predictor/
      part2_m64.sh # 86 configs
8 bash ./scripts/final/3_single_op_with_predictor/
      part2_m4096.sh # 46 configs
9 # monitor processes running in the background
10 ps -aux | grep -c single_op_with_predictor_part2
```

Each script launches one process per config. As the simulation for MVMs (part2\_m1-m64.sh) can be finished quickly, you can start the next script whenever free CPU cores are available. As the simulation for MMs (part2\_m4096.sh) may take few hours to one day, we also provide all the result logs on Zenodo, under flexq\_ndp/ scripts/final/3\_single\_op\_with\_predictor/ log\_rebuttal\_mm\_new.

Step 3: Theoretical Latency Lower Bound (< 1 hour)

<sup>1</sup> bash ./scripts/final/3\_single\_op\_with\_predictor/ mm\_speedup\_optimal.sh

# *E. Experiment Workflow*

Note1: All command lines in Appendix XIII-E should be run from the flexq\_ndp project root (under flexq\_ndp directory). All result paths provided in the workflow is under flexq\_ndp/scripts/final directory.

Note2: Experiment for GEMM operator speedups (Fig. 8) must be completed before running any other experiments, as subsequent steps depend on its output.

*1) GEMM Operator Speedup (Fig. 8):* arranged in four steps, each step must wait for the preceding step is finished (including background processes).

Step 1: Search compilation strategies (∼1 day)

```
1 # run in the background
2 bash ./scripts/final/3_single_op_with_predictor/
     part_1.sh
3 # monitor processes running in the background
4 ps -aux | grep -c single_op_with_predictor_part1
```

Best strategies found during Part 1 are stored in 3\_single\_op\_with\_predictor/ tmp\_strategy/tmp\_strategy.yaml

Step 2: Simulate Baseline & FlexQ-NDP Latency (∼1 day)

```
1 bash ./scripts/final/3_single_op_with_predictor/
      part2_m1.sh # 94 configs
2 bash ./scripts/final/3_single_op_with_predictor/
      part2_m2.sh # 20 configs
3 bash ./scripts/final/3_single_op_with_predictor/
      part2_m4.sh # 86 configs
4 bash ./scripts/final/3_single_op_with_predictor/
      part2_m8.sh # 20 configs
5 bash ./scripts/final/3_single_op_with_predictor/
      part2_m16.sh # 94 configs
6 bash ./scripts/final/3_single_op_with_predictor/
      part2_m32.sh # 20 configs
7 bash ./scripts/final/3_single_op_with_predictor/
      part2_m64.sh # 86 configs
8 bash ./scripts/final/3_single_op_with_predictor/
      part2_m4096.sh # 46 configs
9 # monitor processes running in the background
10 ps -aux | grep -c single_op_with_predictor_part2
```

Each script launches one process per config. As the simulation for MVMs (part2\_m1-m64.sh) can be finished quickly, you can start the next script whenever free CPU cores are available. As the simulation for MMs (part2\_m4096.sh) may take few hours to one day, we also provide all the result logs on Zenodo, under flexq\_ndp/ scripts/final/3\_single\_op\_with\_predictor/ log\_rebuttal\_mm\_new.

Step 3: Theoretical Latency Lower Bound (< 1 hour)

<sup>1</sup> bash ./scripts/final/3\_single\_op\_with\_predictor/ mm\_speedup\_optimal.sh

