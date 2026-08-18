# *A. Limitation of Gust-based Dataflow in Large-scale Designs*

Gustavson dataflow introduces irregular accesses to Matrix B, and processing different elements of Matrix A in parallel results in concurrent random accesses. Although many Gustbased accelerators address this challenge through cache-based optimization techniques, they are typically implemented with a small number of MAC units (16-64) [19], [22], [24]. While these techniques are effective for small designs, they may not scale well, as cache complexity and contention grow significantly with higher degrees of parallelism in larger systems.

Trapezoid [41] is a large-scale Gust-based accelerator with 128×128 MAC units. To mitigate the concurrent irregular accesses at a large scale, it adopts a multi-level memory hierarchy. The MAC units are divided into four clusters (32 PE rows per cluster), each sharing a 4MB, 32-bank cache, connected via a 32×32 crossbar. However, for HS×HS workloads, Trapezoid obtains only 512 MACs/cycle despite a peak of 16K MACs/cycle. To study its memory behavior, we model the cluster-level memory system as a 32-bank SRAM (1 R/W per bank per cycle), consistent with the reported design.

As shown in Fig. 4(a), speedup initially improves with increasing PE rows but quickly saturates and MAC utilization drops to around 10% at 256 rows. The decline stems from bank conflicts that intensify as the number of concurrent requests increases, introducing memory stalls that limit effective compute throughput. This contributes to the limited sustained throughput of Trapezoid on HS workloads. *At a large scale, for Gust-based sparse accelerators, the memory subsystem, not the compute resources, becomes the dominant bottleneck.* Although increasing the number of memory banks can reduce conflict probability, it incurs substantial area and power overhead due to the quadratic growth of the crossbar.

![](_page_3_Figure_0.jpeg)

\* A total of 10 non-zero multiplications in this sparse matrix example. Fig. 5: An example comparing the execution of HS×HS

on Trapezoid and HiT. Trapezoid incurs one bank conflict due to a concurrent request to Bank 1, executing 1 out of 4 multiplications in this illustrated step. HiT streams B rows from different banks without contention, executing all 4 multiplications in this step. In practice, HiT processes 4 elements of A in parallel, matching multiple consecutive rows of B.

