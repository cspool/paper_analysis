# B. Methodology

To demonstrate the performance gain and memory efficiency, we compare MPipeMoE against the state-of-art system FasterMoE [22], which implements dynamic shadowing and pipeline parallelism in MoE training. We also choose FastMoE as another competitor, which implements the primitive expert parallelism without pipeline parallelism.

We implement *MPipeMoE* and its variant *PipeMoE* to demonstrate the advantages of adaptive pipeline parallelism and memory efficiency. *PipeMoE* implements micro-batch size splitting, which also adopts multi CUDA streams to execute computation and communication in parallel. *MPipeMoE* is implemented on top of *PipeMoE*, which adopts adaptive memory reusing strategies to further reduce memory footprint.

