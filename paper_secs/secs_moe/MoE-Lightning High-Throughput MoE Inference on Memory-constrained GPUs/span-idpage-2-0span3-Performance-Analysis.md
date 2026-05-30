# <span id="page-2-0"></span>3 Performance Analysis

In this section, we introduce a Hierarchical Roofline Model (HRM) (§3.2) extended from the classical Roofline Model [48], which we use to conduct a theoretical performance analysis for MoE inference (§3.3). It also serves as the basics of our performance model used in scheduling policy search, which will be discussed in §4.2. The Hierarchical Roofline Model extends the original Roofline Model for multicore architectures [48] to provide a stronger model of heterogeneous computing devices and memory bandwidth. We further identify two additional turning points that define settings where the computation is best done on CPU instead of GPU and where the application is GPU memory-bound or CPU memory-bound, providing explicit explanations for how LLM inference performance will be affected by different resource limits in the system.

#### 3.1 Roofline Model

We will start with the original Roofline Model [48], which provides a visual performance model to estimate the performance of a given application by showing inherent hardware limitations and potential opportunities for optimizations. It correlates a system's peak performance and memory bandwidth with the operational intensity of a given computation, where Operational Intensity (*I*) denotes the ratio of the number of operations in FLOPs performed to the number of bytes accessed from memory, expressed in FLOPs/Bytes.

The fundamental representation in the Roofline Model is a performance graph, where the x-axis represents operational intensity *I* in FLOPs/byte and the y-axis represents perfor-

![](_page_2_Figure_12.jpeg)

mance *P* in FLOPs/sec. The model is graphically depicted by two main components:

**Memory Roof:** It serves as the upper-performance limit indicated by memory bandwidth. It is determined by the product of the peak memory bandwidth ( $B_{\text{peak}}$  in Bytes/sec) and the operational intensity (I). Intuitively, if the data needed for the computation is supplied slower than the computation itself, the processor will idly wait for data, making memory bandwidth the primary bottleneck. The memory-bound

region (in blue) of the roofline is then represented by:

$$P \le B_{\text{peak}} \times I$$
 (1)

where P is the achievable performance.

**Compute Roof:** This represents the maximum performance achievable limited by the machine's peak computational capability ( $P_{\text{peak}}$ ). It is a horizontal line on the graph (top edge of the yellow region), independent of the operational intensity, indicating that when data transfer is not the bottleneck, the maximum achievable performance is determined by the processor's computation capability. The compute-bound part (yellow region) is then defined by:

$$P \le P_{\text{peak}}$$
 (2)

The turning point is the intersection of the compute and memory roofs, given by the equation:

<span id="page-3-5"></span>
$$\bar{I} = \frac{P_{\text{peak}}}{B_{\text{peak}}} \tag{3}$$

defines the critical operational intensity  $\bar{I}$ . Applications with  $I \geq \bar{I}$  are typically *compute-bound*, while those with  $I < \bar{I}$  are *memory-bound*.

In practice, analyzing an application's placement on the roofline model helps identify the critical bottleneck for performance improvements. Recent works [52] analyze different computations (e.g., softmax and linear projection) in LLM using the Roofline Model.

#### <span id="page-3-0"></span>3.2 Hierarchical Roofline Model

While the original Roofline Model demonstrates great power for application performance analysis, it is not enough for analyzing applications such as LLM inference that utilize diverse computing resources (e.g., CPU and GPU) and move data across multiple memory hierarchies (e.g., GPU HBM, CPU DRAM, and Disk storage).

Consider a system with n levels of memory hierarchies. Each level i in this hierarchy is coupled with a computing processor. The peak bandwidth at which the processor at level i can access the memory at the same level is denoted by  $B_{\rm peak}^i$ . Additionally, the peak performance of the processor is denoted by  $P_{\rm peak}^i$ .

**Definition 3.1** (General Operational Intensity). To consider different memory hierarchies, we define the general operational intensity  $I_x^i$  of the computation task x as the ratio of the number of operations in FLOPs performed by x to the number of bytes accessed from memory at level i.

For computation x executed at level i in the HRM, we can define its compute and memory roofs similarly as in the original Roofline Model:

#### • Compute Roof at level i:

<span id="page-3-2"></span>
$$P_x^i \le P_{\text{peak}}^i \tag{4}$$

This represents the maximum computational capability at level *i*, independent of the operational intensity.

## • Memory Roof at level i:

$$P_x^i \le B_{\text{peak}}^i \times I_x^i \tag{5}$$

More importantly, in HRM, there is also the memory bandwidth from level j to level i, denoted as  $B_{\text{peak}}^{j,i}$ , which will define another memory roof for computation x that is executed on level i and transfers data from level j:

