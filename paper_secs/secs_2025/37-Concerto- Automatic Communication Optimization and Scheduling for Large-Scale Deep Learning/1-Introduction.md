# 1 Introduction

With the rapid development of deep learning (DL), there is a growing demand for scale. To pursue model accuracy in various tasks, including computer vision (CV) [\[11,](#page-14-0) [14,](#page-14-1) [29\]](#page-15-1) and natural language processing (NLP) [\[5,](#page-14-2) [12,](#page-14-3) [43\]](#page-15-2), increasingly massive models are being proposed. However, training these advanced models requires numerous GPU resources. In particular, training large language models involves the supercomputer composed of thousands or even tens of thousands of GPUs [\[31,](#page-15-3) [41\]](#page-15-4). Nevertheless, such massive training scales correspond to significant time, economic, and environmental costs. To promote the development of DL technology

and ensure environmental sustainability [\[3\]](#page-14-4), enhancing hardware utilization efficiency and reducing training time have become crucial topics.

In large-scale DL training, we need to jointly use different parallelism approaches which will introduce communications. These communications can become bottlenecks in training and impede the efficiency of scaling. Optimizing these communications becomes a fundamental need. Despite significant advancements in communication hardware capabilities, such as NVLink and InfiniBand, the time spent on communication during training remains a bottleneck. The proportion of time dedicated to communication may account for 20% to 40% of the total training duration on modern clusters [\[47\]](#page-15-5).

To mitigate communication overhead, various optimization solutions have been proposed. For asynchronous communications, system researchers and developers are required to implement specific scheduling in specific scenarios so that they can effectively overlap with computation and efficiently utilize network bandwidth. For example, PyTorch Distributed Data Parallel (DDP) [\[27\]](#page-14-5) organizes parameter gradients into buckets and kicks off asynchronous all-reduce per bucket. For tensor parallelism which introduce synchronous all-reduce. Megatron-LM [\[31\]](#page-15-3) v2.7 introduce asynchronous all-reduce in backward of linear layer which significantly reduces the cost of tensor parallelism communication in the back-propagation. For critical (synchronous) communications, some work [\[47\]](#page-15-5) has proposed that the contextual computation can be decomposed to achieve overlap.

However, these manual optimizations introduce the following challenges:

Challenge 1: Performance - These manual optimizations, which do not fully exploit the opportunity of overlapping communication computations, leaving room for improvement. Moreover, some communication optimizations contain empirical parameters, such as the need to set bucket size in PyTorch DDP, and the default values of these parameters may be inappropriate in varied scenarios.

Challenge 2: Programmability - Performing communication optimization manually requires the developer to manage asynchronous communication, including control synchronization, and communication fusion, which are nontrivial and increase the complexity of the system. Furthermore, these optimizations are implemented in PyTorch's eager mode by re-implementing models or optimizers that hard to be integrated into the PyTorch compiler stack.

Challenge 3: Generality - Currently these communication optimization efforts are intertwined in the implementation of parallel approaches. It is exceedingly difficult to apply existing communication optimizations to more complex or new parallel approaches. For instance, in auto-parallelism, where the parallelism and the communication pattern are uncertain, predefined scheduling and optimization approaches cannot be utilized. Additionally, the current optimizations

for critical communication (decomposition) are specific to Transformer [\[45\]](#page-15-6) and cannot be generalized to arbitrary models. As of now, there is no system that can generally optimize communication for arbitrary parallelism approaches.

To address these challenges, we propose Concerto, a compiler framework for automatic optimization and scheduling of communication. We abstract communication optimization as a resource constrained project scheduling problem (RCPSP). Through off-the-shelf solver, Concerto can generate optimized topological sorting. Furthermore, Concerto introduces auto-decomposition to create optimization space for critical communication.

In summary, we make the following contributions:

- We propose Concerto, a compiler framework for automatic optimization and scheduling of communication, tailored for various models across different parallelization approaches.
- We formulate the scheduling problem as a resource constrained project scheduling problem and use offthe-shelf solver to get the near-optimal scheduling. And use auto-decomposition to create overlap opportunity for critical (synchronous) communication.
- We implement Concerto with PyTorch 2.0 [\[2\]](#page-13-0) compiler stack and provide users with the one-line API for parallelism and communication optimization.
- We evaluate Concerto with the state-of-the-art distributed training frameworks such as Megatron-LM [\[31\]](#page-15-3), Jax/XLA [\[17\]](#page-14-6), DeepSpeed [\[38\]](#page-15-7) and Alpa [\[52\]](#page-15-8). For PTD parallelism, Concerto can match the highly optimized system Megatron-LM and Jax/XLA. Concerto accelerates Evoformer by up to 19.7% with dynamic axial parallelism. For ZeRO-powered data parallelism, compared with DeepSpeed, Concerto achieves maximum performance improvement of 42.9% and an average improvement of 19.1%. For automatic parallelism, Concerto achieves 22.7% maximum and averaging 11.1% compared with Alpa.

