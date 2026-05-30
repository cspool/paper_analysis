# *D. Expert Parallelism*

Compared to traditional Transformer models of the same model capacity, MoE models offer an interesting trade-off. MoE requires much less computation but much more memory usage. Expert layers deploy many additional FFNs, which increase model size and associated demands for memory capacity near the compute device (*e.g.*, the GPU). To handle this problem, GShard [\[21\]](#page-11-1) proposes expert parallelism, which distributes the workload across multiple devices to reduce memory and computation per device.

With expert parallelism, MoE layers are distributed across multiple devices. Each device holds only a subset of expert FFNs and a copy of all the other parameters. When a token is assigned to experts that reside on other devices, an all-toall communication collective sends the token to corresponding devices. The tokens are processed by the expert and then sent back by another all-to-all communication.

At maximum expert parallelism, which allocates one expert per device, memory usage and FLOP count per device are comparable to that from a dense transformer model. Since the gating function is a lightweight linear layer, the overall computational complexity of a batch is about the same as that of a dense transformer with much fewer parameters. Nevertheless, the enormous size, the sparse activation of experts, and the complex communication pattern between devices hosting different experts poses severe challenges during model deployment and inference.

