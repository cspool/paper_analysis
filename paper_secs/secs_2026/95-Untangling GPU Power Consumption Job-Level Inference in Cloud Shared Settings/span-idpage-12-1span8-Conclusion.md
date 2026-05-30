# <span id="page-12-1"></span>8 Conclusion

In this paper, we examined the power consumption of individual jobs in different GPU-sharing configurations. Through empirical benchmarks, we observed that power consumption is linear with load, but also influenced by several key factors: high static power draw, maximum power reached before full utilization (a "last-for-free" effect), and measurement variations due to drivers and monitoring tools. Notably, outdated GPUs drivers (version 535 vs. 570) can nearly double the power use of MIG accelerators, and the default active profiling of the DCGM exporter can similarly increase static power in certain configurations.

We proposed models for estimating power in time-shared settings (load-based), spatially shared settings (slice-sizebased), and passthrough settings (temperature-based, accuracy of 76.7%). These models are tailored for cloud environments operating in a multi-tenant black-box context, preserving workload privacy while enabling integration with cloud providers' carbon accounting tools to help limit power consumption.

Our findings also show that GPU sharing can improve energy efficiency for certain jobs, where reduced per-job power consumption outweighs performance degradation. Small AI-like workloads, in particular, appear to benefit from such sharing strategies.

Applying the passthrough method to an OVHcloud cluster, we found significant underutilization of GPU compute resources in IaaS contexts, reinforcing the potential of accelerator sharing.

Future work will extend these inference techniques to other GPU architectures (e.g., AMD and Intel) and integrate them into energy-aware scheduling strategies.

