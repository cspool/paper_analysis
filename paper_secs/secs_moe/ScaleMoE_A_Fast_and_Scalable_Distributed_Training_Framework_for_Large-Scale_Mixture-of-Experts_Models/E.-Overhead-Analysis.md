# *E. Overhead Analysis*

To clarify the overheads caused by clustering-related operations (i.e., clustering & remapping, clustering for input), we break down the overheads of each operation. Table IV presents the detailed overhead breakdown. Even the most timeconsuming operation, dynamic expert clustering, introduces negligible latency compared to iteration time. Without overlap, the total overhead per iteration is 568.91ms (8.51% of the 6679.27ms). With overlap, it drops to 16.27 ms (0.26%). This shows that the additional overheads introduced by ScaleMoE are negligible and do not impact the overall performance.

