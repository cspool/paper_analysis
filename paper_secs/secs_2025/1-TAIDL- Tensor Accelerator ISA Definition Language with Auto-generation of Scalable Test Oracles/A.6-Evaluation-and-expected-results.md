# A.6 Evaluation and expected results

The key results of the paper include benchmarking simulation times of TAIDL-TOs and baselines (Gemmini Spike and Intel SDE), i.e., statistics reported in Figure [19,](#page-10-2) Figure [20,](#page-11-1) Figure [21,](#page-12-2) and [§8.2.](#page-12-3)

Note that most of the simulation times are in milliseconds, so machine characteristics (processor performance) and runtime characteristics (background activity) may result in numerical variations of final results. Nevertheless, the following trends will be consistent:

- (1) Figure [19:](#page-10-2) Gemmini Spike is expected to be significantly slower than TAIDL-TO with and without GPU acceleration.
- (2) Figure [20:](#page-11-1) Intel SDE is expected to be slower than TAIDL-TO with and without GPU acceleration, with a minor exception for AVX-only kernels on GPU acceleration.
- (3) Figure [21:](#page-12-2) TAIDL-TO is expected to simulate Exo-generated Gemmini kernels within a second per kernel.
- (4) [§8.2:](#page-12-3) TAIDL-TO is expected to be orders of magnitude faster than Gemmini Spike (milliseconds/seconds vs minutes)

