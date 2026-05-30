# F. Comparison with High-Performance System

This section discusses the performance gap between our budget-friendly LLM inference system Hermes and state-of-the-art high-performance serving system TensorRT-LLM [41].

We kept the input and output sequence lengths set at 128. To handle LLaMA2-70B with a batch size of 16, TensorRT-LLM requires five NVIDIA A100-40GB-SXM4 GPUs. In contrast, Hermes operates with only one NVIDIA RTX-4090 GPU and affordable NDP-DIMMs. Figure 17 displays the performance comparison between TensorRT-LLM and Hermes. For a batch size of 1, Hermes achieves 79.1% inference efficiency of TensorRT-LLM. Even at a batch size of 16, Hermes retains 24.4% inference efficiency of TensorRT-LLM. Despite this, Hermes is far more economical than TensorRT-LLM, which is equipped with 5 NVIDIA A100-40GB-SMX4 GPUs. Specifically, Hermes only costs approximately \$2,500, whereas TensorRT-LLM requires \$50000 to support LLaMA2- 70B. Hermes provides efficient and low-budget LLM inference for local deployments.

