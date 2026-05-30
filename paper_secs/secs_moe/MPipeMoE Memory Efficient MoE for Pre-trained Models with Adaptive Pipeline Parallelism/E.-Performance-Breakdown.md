# E. Performance Breakdown

To better understand the performance breakdown of MPipeMoE, we reveal the performance of the different methods in *memory-time* coordinates, in which the x-axis represents the memory footprint and the y-axis represents the training time. As shown in Figure 11, the one closer to the origin point illustrates better overall performance. MPipeMoE significantly outperforms FasterMoE and FastMoE. PipeMoE(n=4) reduces the training time because of a higher GPU throughput. PipeMoE outperforms PipeMoE(n=4) by configuring the optimal pipeline granularity at runtime. MPipeMoE achieves best memory efficiency by reusing memory partitions. The higher GPU utilization makes it possible to increase the batch size with the limited device memory space.

![](_page_8_Figure_8.jpeg)

Fig. 12. The effects of pipeline parallelism on various pipeline granularity. The dashed line represents the adaptive granularity selected by the configuration algorithm. The x-axis represents various B values.

![](_page_8_Figure_10.jpeg)

Fig. 13. The overhead of memory reusing strategies and the effectiveness of the strategy selection method in MPipeMoE. The ticks of the x-axis represent different numbers of GPUs N and the batch size of tokens B in format (N,B).

#### F. Effectiveness of Granularity Configurations

We illustrate the effectiveness of the adaptive pipeline granularity configuration of MPipeMoE, which is based on a hypothesis that n is monotonically increasing as B increases. We compare the performance due to different pipeline granularity with various batch sizes of tokens on model GPT-XL. Figure 12 shows that when the batch size is smaller than 8k, n=2 is the best option. When the batch size is increased to 8k-22k, n=4 ensures the best performance. n=8 is the optimal configuration if the batch size is larger than 22k. MPipeMoE, which is denoted as a dashed line, performs the best in all situations. The results validate its effectiveness.

#### G. Overhead of Memory Reusing

In terms of speedup, MpipeMoE is indeed second to PipeMoE because MpipeMoE achieves memory efficiency at the same time, which however incurs non-trivial overhead. MpipeMoE features four memory reusing strategies, i.e., S1, S2, S3, and S4 defined in Table II, which resort to recomputation/communication and CPU offloading to restore activation tensors in the backward pass. For overhead analysis of the strategies, we conduct experiments with different numbers of GPUs N and various batch sizes of tokens B. Figure 13 presents the results, from which we can observe that:

- S1 and S2 perform better when N is small, e.g., 8, but worse with a larger N, e.g., 64. S1 and S2 introduce additional memory copy operations while S2 introduces additional communication operations. With the increasing number of workers, the cost of communication also increases, which results in the worse performance for S2 due to the competition on the memory bandwidth between memory copy and communication.
- Both S3 and S4 introduce additional computational costs, which perform worse if the workload is computationbound, i.e., N = 8.
- S4 performs better than S2 if N equals 32 or 64, in which communication is the bottleneck because memory copy over PCIe in S2 slows down communication operations.
- There is not much performance variation with the varying batch sizes, indicating that the batch size is not sensitive to the configuration of strategy.

Based on these observations, we can conclude that there does not exist a single memory reusing strategy which can ensure the best performance under all situations. MPipeMoE builds a performance model based on Equation 10 to decide the optimal strategy considering both the hardware configurations and runtime characteristics.

