# <span id="page-17-0"></span>4 RESOURCE-EFFICIENT ALGORITHMS

This section focuses on resoruce-efficient large FMs techniques at the algorithm level. Compared to traditional DNNs, large FMs exhibit new characteristics such as its huge parameter set and autoregressive inference. This disparity has led to the emergence of numerous resource-efficient algorithms, which are categorized based on the lifecycle of FMs: pre-training, fine-tuning, serving algorithms, and model compression as illustrated in Figure 11.

#### <span id="page-17-1"></span>4.1 Pre-training Algorithms

Pre-training for large FMs relies on a substantial amount of computation resources. For instance, GPT-3-175B [\[41\]](#page-39-0) consumes 3.14 × 10<sup>23</sup> flops and LLaMa-70B [\[383\]](#page-58-1) takes 1.7 × 10<sup>6</sup> GPU hours. Consequently, optimizing the utilization of computational resources is crucial for the efficient pre-training of FMs. Resource-efficient algorithms can be categorized into training data deduction, neural architecture search, progressive learning, and mixed precision training.

