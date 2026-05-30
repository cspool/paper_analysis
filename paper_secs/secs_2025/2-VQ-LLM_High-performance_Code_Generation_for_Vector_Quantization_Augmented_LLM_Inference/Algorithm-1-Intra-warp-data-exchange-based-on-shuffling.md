# Algorithm 1 Intra-warp data exchange based on shuffling

```
\overline{\textbf{Input: } data,} iter, layout_{dequant, compute}
Output: data
 1: \  \, \mathbf{function} \  \, \mathsf{THREAD\_Mapping}(data, layout_{dequant, compute})
        for item \in data do
3:
            item.tid_{compute,dequant} \leftarrow GetTid(item, layout_{compute,dequant})
4:
        mini\_warps \leftarrow []
        for dequant\_thread \in warp do
6:
             mw \leftarrow [data.tid_{compute} \ for \ data.tid_{dequant} = dequant\_thread]
7.
             if mw \notin mini\_warps then
8:
                 mini\_warps[mw] \leftarrow []
9.
             mini\_warps[mw].append(dequant\_thread)
         for mw \in mini\_warps do
10:
             mini\_warps[mw][i] \leftarrow mw[i] // Thread mapping we need
11:
12: function Reg_Fusion(data, iter)
         for off in [1, iters) do // intersected 0 no shuffle needed
14:
             data[tid \circ ff] \leftarrow shfl_{xor}(data[tid \circ ff], off)
```

2) Implementation: We outline our algorithm in Alg. 1. To determine the thread mapping, we first find the association between each element in terms of dequantization and computation (lines 2-3). Subsequently, for each thread, we identify all threads that require its dequantized data, grouping these threads into a mini-warp (lines 4-6). We then construct mini-warps for all threads (lines 7-9). In the previous example, threads 0, 1, 16, and 17 possess identical data [0, 1, 2, 3] and thus form a mini-warp. Finally, we remap all threads by miniwarps (lines 10-11); for instance, we assign threads 2 and 3 to dequantize the data initially handled by threads 16 and 17. This process is executed offline to ensure proper thread mapping in runtime dequantization, enabling the implementation of register-level fusion via the shuffle API (lines 12-15).

Adaptivity. Clearly, a larger discrepancy between the dequantization layout and the required layout of the computation kernel increases the need for shuffling. Consequently, we propose conducting hierarchical fusion adapted to the vector size of the codebook entry. Profiling results indicate that the latency of shared memory access is nearly five times that of register access combined with shuffling. Therefore, for quantized tensors requiring fewer than five shuffle operations, we implement register-level fusion. For other tensors, we maintain the conventional shared memory-level fusion.

## C. Overall Workflow

Our compute engine adopts a template-based design in Alg. 2 to generate final fused kernels. First at the offline phase, based on the VQ configuration and targeted computation, we determine shared/register budgets, split factors, required number of shuffles, and the corresponding thread mapping for our proposed optimizations (lines 2-8).

Subsequently, we launch the codebook-centric dataflow computation (line 9) via the **Parallel\_For** function that binds following sub-tasks to parallel thread blocks. Its two parameters represent the task splitting axes and the split factor, respectively. Within each parallelized task, we first load the codebook into the codebook cache (lines 10-12), followed by dequantization using the provided APIs in Sec. V (lines

13-14). Notice now threads are mapped to quantized data following **Thread\_Mapping** determined offline, for minimum shuffle if applicable. After dequantization, we perform codebook-centric hierarchical fusion (lines 15-18) using the **Reg\_Fusion** and **Shared\_Fusion** function. Both functions accept dequantized data, with the former requiring a counter  $n_{shuffle}$  to indicate the number of required shuffle operations and latter requiring the source-destination layout to initialiate correct shared memory accesses. Once the data is in the proper layout, we proceed with computation (lines 19-20). Finally, we perform a global reduction if necessary (line 21) via the **Reduce** function, where the first parameter specifies the partial result to be reduced and the second determines the axes along which the global reduction is conducted.

