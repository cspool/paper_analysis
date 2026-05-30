# <span id="page-2-3"></span>2.2 GPU Pipeline Performance Analysis

A simplified but fundamental way to quantify GPU running time is through

<span id="page-2-5"></span>
$$t = \frac{\#inst}{IPC},\tag{1}$$

where #inst is the number of instructions and IPC indicates the instruction issue rate. While we have not yet covered specific tasks, we can provisionally assume an infinite number of instructions and focus on maximizing IPC. Note that in real world, there still remains significant potential for improvement. Our benchmark reveals that even the best

<span id="page-2-4"></span>![](_page_2_Picture_14.jpeg)

Figure 3: Microarchitecture of NVIDIA A100 Tensor Core GPU. We omit some structures such as convergence barrier units and texture units for simplicity.

hand-optimized large matrix multiplication implementation [\[8\]](#page-14-16) achieves only 3.4 IPC (out of a theoretical maximum of 4) on A100, let alone other workloads and frameworks.

To maximize IPC, we begin with a simple case in which an SMSP manages only one warp. Here, optimizing ILP is the only way to enhance IPC since there are no other warps to hide latency. For better ILP, we need to reduce structural hazards (e.g., math\_pipe\_throttle due to math pipe oversubscription) and data hazards (e.g., stalled\_wait from fixed execution dependencies). The absence of OoOE makes the optimization of ILP heavily rely on compilation techniques such as instruction scheduling and register allocation [\[38\]](#page-14-15).

In real world, modern GPUs utilize multiple warps within the same SMSP to cover each other's instruction latency. However, optimizing both ILP and TLP presents a trade-off in that increasing registers of threads enhances ILP but weakens TLP, and vice versa. An important fact for addressing this trade-off is that excessive TLP is not advantageous. TLP beyond 4 is inefficient due to the high cost of inter-warpgroup switching, as discussed in § [2.1.](#page-2-2) In addition, too many warps can lead to performance issues with cache coherence and data locality. In GPU high-performance practice [\[1\]](#page-14-17), a TLP of 4 effectively covers warp latency. Therefore, we can resolve the ILP-TLP optimization trade-off by

<span id="page-2-6"></span>
$$\max ILP, \quad \text{s.t.} \quad TLP \ge 4. \tag{2}$$

No upper bound for TLP is necessary as maximizing the optimization objective implies minimizing TLP.

