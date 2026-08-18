# VII. LIMITATIONS AND DISCUSSION

NCAs are very effective for fine-grained tasks that are both computationally and memory demanding, and that require tight CPU core control and communication. We demonstrated the NCA effectiveness in popular kernels of important workloads [1], [27]. However, NCAs and ATX are not intended to replace ICAs or OCAs for all accelerated code. In this section, we discuss the limitations of NCAs when compared to them.

OCAs have some advantages over NCAs. In contrast to OCAs, NCAs cannot write to the memory system, a design decision that (1) limits the output size of the task to register sizes, and (2) does not improve the performance of memory stores over general-purpose cores. The first limitation can be largely alleviated by using the large register sizes provided by current processors (e.g., 1 KB tile registers) and by the fact that, often, tasks can be scaled to the desired size (e.g., with matrix tiling). The second limitation is not crucial for many workloads because stores are generally less critical than loads for performance [\[46\]](#page-14-25), [\[75\]](#page-15-11). Still, for coarse-grained writeintensive operations such as memory scatters [\[54\]](#page-14-26) or memory copies, OCAs may be the better option. Another advantage of OCAs is that they can be shared between cores, whereas NCAs are private to each core.

In some applications, ICAs may be the best option. NCAs excel at fine-grained tasks whose input data is stored in memory. However, ICAs may be better for tasks where most input data can be stored in registers and the L1 cache.

Overall, NCAs are an additional design point between ICAs and OCAs, addressing a gap in terms of core-accelerator communication overhead and memory access efficiency. A full analysis and partitioning of tasks between ICAs, NCAs, and OCAs is an open problem and our future work.

# VII. LIMITATIONS AND DISCUSSION

NCAs are very effective for fine-grained tasks that are both computationally and memory demanding, and that require tight CPU core control and communication. We demonstrated the NCA effectiveness in popular kernels of important workloads [1], [27]. However, NCAs and ATX are not intended to replace ICAs or OCAs for all accelerated code. In this section, we discuss the limitations of NCAs when compared to them.

OCAs have some advantages over NCAs. In contrast to OCAs, NCAs cannot write to the memory system, a design decision that (1) limits the output size of the task to register sizes, and (2) does not improve the performance of memory stores over general-purpose cores. The first limitation can be largely alleviated by using the large register sizes provided by current processors (e.g., 1 KB tile registers) and by the fact that, often, tasks can be scaled to the desired size (e.g., with matrix tiling). The second limitation is not crucial for many workloads because stores are generally less critical than loads for performance [\[46\]](#page-14-25), [\[75\]](#page-15-11). Still, for coarse-grained writeintensive operations such as memory scatters [\[54\]](#page-14-26) or memory copies, OCAs may be the better option. Another advantage of OCAs is that they can be shared between cores, whereas NCAs are private to each core.

In some applications, ICAs may be the best option. NCAs excel at fine-grained tasks whose input data is stored in memory. However, ICAs may be better for tasks where most input data can be stored in registers and the L1 cache.

Overall, NCAs are an additional design point between ICAs and OCAs, addressing a gap in terms of core-accelerator communication overhead and memory access efficiency. A full analysis and partitioning of tasks between ICAs, NCAs, and OCAs is an open problem and our future work.

