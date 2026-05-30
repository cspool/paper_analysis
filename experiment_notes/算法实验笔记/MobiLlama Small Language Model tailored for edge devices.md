## MobiLlama Small Language Model tailored for edge devices

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MobiLlama 是一种 0.5B（及 0.8B）参数的 SLM（Small Language Model），核心创新在于**共享 FFN（Feed-Forward Network）设计**：不同于传统 Transformer 每层一个独立 MLP 块（含多个 FFN 层），MobiLlama 让所有 22 层 Transformer block 共享同一个 MLP 块。在常规设计中（如 large-base），FFN 层占 65% 的总参数量，通过共享 FFN 可将参数量减少约 60%，使得在保持 22 层 + hidden dim 2048 的高容量配置下，总参数仅 0.5B（与低容量 baseline 相同）。0.8B 版本通过拓宽共享 FFN（hidden dim 2532, intermediate 11080）获得更高精度。
  - 实验比较：
    - **Baseline 对比（同训练budget）**：baseline1（22 层/1024 hidden/0.54B）、baseline2（8 层/2048 hidden/0.52B）、large-base（22 层/2048 hidden/1.2B）在 100B tokens 上预训练，4 benchmarks（HellaSwag, TruthfulQA, MMLU, Arc_C）对比。
    - **SOTA 对比**：与 gpt-neo-125m、tiny-starcoder、cerebras-gpt-256m、opt-350m、megatron-gpt2-345m、LiteLlama、gpt-sw3-356m、pythia-410m、xglm-564m、Lamini-GPT-LM 等 <1B 模型在 9 benchmarks 对比。
    - **Efficiency 对比**：在 RTX2080Ti GPU（bf16）、i7 CPU（4bit GGUF）、Snapdragon-685 手机（4bit GGUF）三个平台上对比 Llama2 7B、Phi2 2.7B、large-base 1.2B，指标含 Avg Tokens/Sec、Avg Memory Consumption、Avg Battery Consumption。
    - **Slicing 对比**：与 SliceGPT 30% 参数 sliced 的 OPT-1.3B/6.7B、Llama-2-7B、Phi2-2.7B 在 4 benchmarks 对比。
    - **多模态评估**：MobiLlama-V 0.8B（CLIP+LLM）在 GQA、SQA、TextQA、MME 上评估。

- 硬件平台是什么，配置是什么。
  - 预训练：20 个 GPU 节点，每节点 8×NVIDIA A100（80GB），800 Gbps 互联，NVLink + 2 port 200 Gb/s (4× HDR) InfiniBand。吞吐约 14k-15k tokens/sec/GPU。
  - 部署测试：PC with RTX 2080Ti GPU（bf16 部署）、Laptop with i7 CPU（4bit GGUF）、Smartphone with Snapdragon-685 processor（4bit GGUF）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：MobiLlama 0.5B（hidden size 2048, 22 layers, 32 heads, intermediate 5632, max seq len 2048, vocab 32000, RMSNorm, RoPE, SwiGLU）；MobiLlama 0.8B（hidden size 2532, intermediate 11080, 其余同 0.5B）；large-base 1.2B（22 layers, hidden 2048, 独立 FFN per layer）。
  - 数据集：预训练用 LLM360 Amber dataset（1.2T tokens），含 Arxiv 30B、Book 28.9B、C4 197.7B、Refined-Web 665B、StarCoder 291.9B、StackExchange 21.8B、Wikipedia 23.9B。
  - Benchmark：HellaSwag（10-shot）、TruthfulQA（0-shot）、MMLU（5-shot）、Arc_Challenge（25-shot）、CrowsPairs（0-shot）、PIQA（0-shot）、Race（0-shot）、SIQA（0-shot）、Winogrande（5-shot），共 9 个 benchmarks。多模态：GQA、SQA、TextQA、MME。
  - 评估框架：Analysis-360 framework（基于 lm-evaluation-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/mbzuai-oryx/MobiLlama（完整训练数据pipeline、训练代码、模型权重、300+ checkpoints、评估代码）
  - 算法 pipeline 核心（共享 FFN 设计）：
    ```
    # 传统 Transformer (如 large-base):
    for layer in 1..L:
        # 每层有独立的 MHA + MLP
        h = h + MHA(LayerNorm(h), layer_id=layer)       # 独立 attention
        h = h + MLP[layer](LayerNorm(h))                 # 独立 FFN，MLP[layer] 是第 layer 层的参数
        # MLP 通常含 3 个 FFN (gate, up, down): W_gate, W_up, W_down

    # MobiLlama (共享 FFN):
    shared_MLP = MLP(W_gate_shared, W_up_shared, W_down_shared)  # 仅一份 FFN 参数
    for layer in 1..L:
        h = h + MHA(LayerNorm(h), layer_id=layer)       # 每层独立 attention（含 Q/K/V/O proj）
        h = h + shared_MLP(LayerNorm(h))                # 所有层共享同一 MLP
    ```
  - 参数量分析：在 large-base 中，FFN 参数占 65%（W_gate/W_up/W_down），attention 占 30%（Q/K/V/O proj），heads 占 5%。通过共享 FFN，整体参数从 1.2B 降至 0.5B（减少约 60%）。
  - 训练超参数：AdamW（β1=0.9, β2=0.95），初始 LR=3e-4，cosine schedule 衰减至 3e-5，weight decay=0.1，gradient clipping=1.0，warmup=2000 steps，batch size=800（160×5），Flash-Attention 加速。
  - 0.8B 版本：在 0.5B 基础上 widening shared FFN（hidden dim 2048→2532, intermediate 5632→11080），其余架构不变。
  - 多模态 MobiLlama-V：CLIP visual encoder 桥接 MobiLlama decoder，在 665k vision-language instruction 数据上端到端微调。
