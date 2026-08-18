# *H. Overhead Analysis*

Preprocessing Overhead. As shown in Figure 16(a), TensorPrism increases modest preprocessing overhead compared with SPADE, HotTiles, and GSpTC by 8.0%, 6.7%, and 4.2%. TCP exhibits the highest preprocessing overhead (25.4%) among all baselines, because, before execution, it explores a large number of possible implementation choices (e.g., tile

![](_page_12_Figure_7.jpeg)

Fig. 17. Area breakdown of TensorPrism Accelerator. sizes and operation kernels). The low cost of TensorPrism

highlights the practicality of incorporating the co-occurrence graph into tensor contraction.

Graph Storage Overhead. To eliminate the impact of on-chip memory size limitations, we evaluate the memory footprint of the input metadata, as illustrated in Figure 16(b). CoG incurs a modest overhead compared with hypergraph representation, with increases of 3.0% on average. Importantly, our primary optimization target is the dense tensor input rather than sparse metadata. Therefore, this slight increase in sparse tensor memory consumption does not constitute a bottleneck and has a negligible impact on overall system efficiency.

## *I. Area Breakdown*

Figure 17 presents the area breakdown of TensorPrism's accelerator. At the accelerator level, the GLB consumes 50% of the total area, and the PE Array takes 46.6% of the total accelerator area. Within each PE, the Contraction Engine dominates at 86.2%, which includes 77.4% of the area allocated to local buffers and registers and 15.1% of the area for MACs. These ratios reflect the design choice to provision substantial local storage per PE, enabling the graph-based dataflow to exploit temporal reuse without frequent GLB accesses. The cooccurrence graph scheduler only incurs 1.9% overhead of total area, in which the cost analyzer takes 62.9% of the scheduler area for partition metadata management.

