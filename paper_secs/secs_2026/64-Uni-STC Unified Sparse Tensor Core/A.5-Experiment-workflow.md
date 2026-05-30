# A.5 Experiment workflow

We provide a unified automation tool qrun to manage experiments. All commands should be executed in the /root/Sim directory:

```
(container)$ cd /root/Sim
```

*Note on Pre-computed Results.* To enable rapid inspection, we have pre-packaged execution logs and generated figures. This allows the subsequent verification instructions to complete in under 10 minutes.

If you prefer to execute the full simulation from scratch to verify the functional reproduction, please clean the pre-existing data using the following commands:

```
# remove figures and execution logs
(container)$ rm /root/Sim/fig/*
(container)$ cd /root/Sim/dist && rm transformer*.csv
spmv/*.csv spmm/*.csv spmspv/*.csv spgemm/*.csv ai/*
(container)$ cd /root/Sim && rm resnet50/dense/*.csv
reset50/sparse/*.csv
```

<sup>1</sup> https://drive.google.com/file/d/1o\_ pdtPdox7aEdRE2e4GtbEPiMFGpPHCu

<sup>2</sup> https://drive.google.com/file/d/ 1Pp3BBOvU8nGoB12bb4o3wZs41twiXwXM

<sup>3</sup> https://drive.google.com/file/d/1o\_ pdtPdox7aEdRE2e4GtbEPiMFGpPHCu

#### A.5.1 Part 1: Fast Verification (L1)

*Estimated Time:* ∼*5 hours — Storage: No extra storage required.* This mode uses small-scale datasets included in the image to reproduce key figures (Fig. 15–19, 21).

• Task 1.1: Format Overhead (Fig. 15)

```
(container)$ qrun format
```

*Explanation:* Evaluates the storage compression ratio of the BBC format across varying sparsity levels.

• Task 1.2: Hardware Comparison (Fig. 17, 18, 19)

```
(container)$ qrun run-sample
```

*Explanation:* Runs SpMV, SpMSpV, SpMM and SpGEMM kernels on representative matrices. Measures performance and energy.

• Task 1.3: Random SpGEMM Evaluation (Fig. 16)

```
(container)$ qrun spgemm2
```

• Task 1.4: AMG Application (Fig. 21)

```
(container)$ qrun run-amg
```

## A.5.2 Part 2: Complete Verification (L2)

*Estimated Time:* ∼*75 hours — Storage:* ∼*500GB required.* This mode downloads the full SuiteSparse collection<sup>4</sup> to reproduce the remaining distribution figures (Figures 20 and 22).

Step 1: Mount Dataset. Download matrix.7z on your host machine, copy it to the container, and extract it.

```
# On Host Machine
$ docker cp matrix.7z HPCA-Pap313:/root
# On Container
(container)$ cd /root
(container)$ 7zz x matrix.7z
(container)$ mv matrix/* /matrix
```

Step 2: Execution.

• Task 2.1: Full Dataset Distribution (Fig. 20)

```
(container)$ qrun run-all # Takes ~24 hours
```

• Task 2.2: Energy Efficiency Density (Fig. 22)

```
(container)$ qrun eed # Takes ~48 hours
```

