# <span id="page-12-0"></span>7 Discussion

Potential for cloud providers. Cloud providers typically expose carbon-footprint accounting for instances through dedicated tools. Integrating our models would improve their accuracy for jobs running under different GPU allocation paradigms, leading to more precise environmental impact assessments. In practice, the power consumption of temporally shared allocations can be modeled based on oversubscription levels, while spatial sharing can be estimated from allocation size. Moreover, temperature serves as a reliable proxy in environments without access to driver-level metrics.

Our results also show that some jobs become more energyefficient under shared settings, supporting the expansion of GPU-shared instance types or the development of managed services designed around them.

Finally, our experiments highlight the influence of drivers and monitoring tools on power consumption. Specifically, using outdated drivers (e.g., 535.X.X) on MIG devices can almost double the power consumption of small slices compared to the latest drivers (at the time of writing, 570.X.X). The DCGM Prometheus exporter can also double the static power consumption on some accelerators if active profiling is used (which is the default option). These are factors that providers should carefully validate before deployment in production environments.

Potential for cloud clients. Selecting an appropriate instance remains the client's responsibility. Our results show that high-end GPUs are often underutilized in dedicated IaaS offers and that jobs can exhibit markedly different energy efficiency depending on the sharing configuration. This suggests that clients could benefit from monitoring the performance and energy footprint of their workloads to guide instance selection. In particular, shared-GPUs represent a viable option to improve efficiency and reduce costs when full accelerator capacity is not required.

Limits. Our approach has practical constraints inherent to the black-box nature of cloud computing and vendorspecific features. The passthrough method requires per-machine calibration and remains sensitive to environmental variation. The time-shared approach could generalize to other accelerators, but spatial sharing is currently tied to NVIDIA MIG feature.

