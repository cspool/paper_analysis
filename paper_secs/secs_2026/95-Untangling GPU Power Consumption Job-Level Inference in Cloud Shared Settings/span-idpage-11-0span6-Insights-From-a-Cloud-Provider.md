# <span id="page-11-0"></span>6 Insights From a Cloud Provider

To illustrate the importance of GPU usage monitoring in cloud computing, we analyze a real GPU cluster from OVH-cloud, composed of 176 H100 GPUs operating in a commercial IaaS environment. To the best of our knowledge, this is the first study to examine a rented GPU cluster in an IaaS context. We use temperature as a proxy to infer the usage of the water-cooled devices in Subsection 6.1, before exploring usage prediction opportunities in Subsection 6.2.

#### <span id="page-11-1"></span>6.1 GPU Usage in an IaaS Cluster

Our analysis, conducted over a two-month period, covered 865 VMs. On average, a VM was active for 2h30 and allocated 1.4 GPUs (with possible values being 1, 2, or 4 on the explored product). Figure 13 shows the utilization of GPUs allocated to VMs. The CDFs highlight a sub-optimal use of the allocated accelerators: 80% of VMs exhibit a median GPU compute usage below 25%. Even when considering peak usage (95th percentile), 60% of deployments remain below 50% of the GPU's compute capacity.

Two main factors influence these observations. First, our definition of GPU utilization is based on the full usage of CUDA cores, as our model was trained using a precise number of cores operating at full capacity, leveraging both GPU-burn and MIG functionalities. In contrast, NVIDIA defines

<span id="page-11-4"></span>![](_page_11_Figure_11.jpeg)

Figure 14. Spearman correlation between different VMs characteristics

utilization more loosely as the "percent of time over the past sample period during which one or more kernels were executing on the GPU" [44]. Under our definition, executing a kernel with a single thread would result in a very low usage estimate, whereas NVIDIA might report the device as fully utilized.

Second, the temperature readings were collected at a 3-minute interval. As a result, short-lived peaks in utilization may not be captured. This is acceptable given that these highend GPUs are primarily intended for long-running jobs.

In practice, most workloads do not fully exploit the device's compute resources due to setup phases, memory-bound behavior, small problem sizes (in terms of CUDA threads), and similar constraints. Altogether, these findings highlight the potential for sharing GPU compute capacity—an opportunity that can be explored through both temporal sharing (Section 3) and spatial sharing (Section 4).

#### <span id="page-11-2"></span>6.2 On GPU Usage Prediction

We now investigate whether GPU compute usage can be predicted from other VM attributes. To this end, we selected several features—VM lifetime, CPU usage, and network activity—and computed their Spearman correlation with three GPU-related metrics: GPU allocation, GPU usage, and peak GPU usage (95th percentile). Results can be seen in Figure 14.

CPU usage does not exhibit strong correlation with GPU usage. We believe this is influenced by the number of *virtual CPUs* (vCPUs) assigned to the VM (starting at 30), which may reduce the statistical visibility of CPU-GPU interactions when workloads are bound to a subset of cores.

Similarly, network activity shows little correlation with GPU usage. While significant data transfers may occur during model loading or checkpointing, we hypothesize that inference workloads are more memory-bound than computebound, limiting network influence on our compute usage estimation. Although training workloads are theoretically compute-intensive, they may also exhibit idle phases due to data movement bottlenecks, further weakening the correlation.

A weak correlation is observed between the number of allocated GPUs and the lifetime of a VM, suggesting that instances with larger GPU allocations tend to run for slightly longer durations.

Overall, these findings highlight the limitations of relying solely on conventional VM metrics to estimate GPU activity in a black-box IaaS context. They reinforce the relevance of alternative proxies such as temperature signals explored in Section [5.](#page-7-0)

