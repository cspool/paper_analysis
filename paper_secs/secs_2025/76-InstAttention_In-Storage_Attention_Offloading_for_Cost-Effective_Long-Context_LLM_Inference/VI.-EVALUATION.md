# VI. EVALUATION

## *A. Methodology*

Inference Systems Setup. We set seven LLM inference systems to thoroughly evaluate the performance of InstAttention over the current KV cache offloading systems:

- 1) DeepSpeed: DeepSpeed-MII system [21] with Zero-Inference [6]. It represents the latest memory-only KV cache offloading system.
- 2) FlexGen: FlexGen system [57], which represents the latest KV cache offloading system to both host memory and SSD for throughput-oriented scenarios. We configure its offload target to SSD to evaluate the SSD-based offloading scheme.
- 3) FlexGen-GDS: FlexGen with GPUDirect Storage [48];
- 4) FlexGen-SparQ: FlexGen with SparQ Attention for sparsity with 1/8 compression ratio;
- 5) Recomp: vLLM system [32], with recomputation for KV cache when it exceeds the available memory;
- 6) InstA: our baseline InstAttention implementation without the SparF Attention mechanism;
- 7) InstA-SparF: complete InstAttention with SparF Attention for sparsity with 1/8 compression ratio;

Testbed Configuration. We conduct our experiments in single CPU-GPU systems. We use NVIDIA A6000 GPU with 48GB VRAM, the 2.2GHz Intel Xeon 5320 CPU with 96GB DDR4 memory, and Samsung 980pro SSDs with 2TB storage. The GPU is connected to CPU via PCIe Gen4x16 lanes.

Model and Datasets. We evaluate OPT-13B, OPT-30B and Llama-2-13B models, which are representative mid-sized LLMs for resource-constrained scenarios. We extend the original FlexGen to support the latest Llama-2-series models. To accommodate all the parameters of OPT-30B, we use two A6000 GPUs for evaluation. We use FP16 for all variables. The sequences for inference are sampled from popular datasets (i.e., ShareGPT [56], Wiki-Text-2 [45], SQuAD [59], and TriviaQA [27]). For OPT models, both the input and output sequence lengths are set to 1024, while for Llama-2 models they are set to 2048. The configuration matches the maximal context length of OPT and Llama-2 models to fully demonstrate the long-context scenarios with heavy KV cache burden.

