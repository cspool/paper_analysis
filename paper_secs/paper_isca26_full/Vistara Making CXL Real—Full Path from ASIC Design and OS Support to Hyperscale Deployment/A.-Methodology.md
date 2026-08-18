# *A. Methodology*

All production experiments are conducted on MemServer nodes. This is a single-socket AMD Turin machine (158 cores, 2-way SMT, ≈3 GHz) with 768 GB local DDR5-6400 and 256 GB CXL-attached DDR4-2400 (2× Vistara ASICs), totaling 1 TB per server. The configuration is shown in Table VI.

We evaluate each workload via A/B testing: identically configured MemServer nodes serve production traffic, differing only in whether CXL memory is enabled (via the cpuset.mems cgroup controller). Traffic is split evenly across test and control groups by the production load balancer, ensuring both groups observe the same request mix and diurnal patterns. Each experiment runs for at least one week to capture steady-state behavior and traffic variability.

We measure service-level metrics (throughput/QPS, p50/p99 latency, cache hit rates, OOM rates) alongside system-level metrics (local and CXL memory bandwidth, CPU utilization, TPP promotion/demotion rates, NUMA hint faults). Table VIII provides a unified view of the TPP telemetry.

