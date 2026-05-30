# 6 Limitations

While our proposed MoDE architecture demonstrates promising results in multi-task LLM adaptation, there are several limitations that warrant further investigation.

Routing Strategy The current MoDE implementation utilizes a relatively simple routing mechanism based on a softmax function. While effective in our experiments, exploring more sophisticated routing strategies that incorporate task relationships or input-specific features could potentially further improve performance.

Hyperparameter Sensitivity The optimal number of experts and rank of the LoRA matrices can vary depending on the specific task distribution and available resources. While our ablation study provides some insights, a more comprehensive exploration of hyperparameter sensitivity could help identify optimal configurations for different scenarios.

Computational Overhead While MoDE significantly reduces parameter count compared to traditional LoRA-MoE, the routing mechanism introduces additional computational overhead during inference. This overhead could become a bottleneck in real-time applications with strict latency requirements. Investigating ways to optimize the routing process or reduce its computational cost would be beneficial.

Evaluation Benchmark Our evaluation primarily focuses on the Supernatural Instructions benchmark. While this dataset covers a wide range of tasks, it may not fully represent the diversity of real-world applications. Evaluating MoDE on other multi-task benchmarks or in specific domains could further validate its effectiveness and generalizability. Addressing these limitations could lead to even more efficient and adaptable multi-task LLM architectures, further expanding the potential of parameter-efficient fine-tuning for a wider range of applications.

