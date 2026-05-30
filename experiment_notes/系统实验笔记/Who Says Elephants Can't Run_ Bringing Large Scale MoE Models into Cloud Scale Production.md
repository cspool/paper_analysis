## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：(1) **扩展 NVIDIA FasterTransformer 推理框架以支持 DeepSpeed MoE 模型**——在 FasterTransformer 中添加 MoE encoder/decoder layer 支持、TUPE attention、gate routing 和 MoE 层的高效计算。(2) **MoE Batch Pruning**——在 decoder 自回归生成过程中动态移除已完成翻译的句子，修改 gating 函数为已完成的句子分配大的 expert_idx（路由到末尾），仅处理 active_tokens 行，避免加载已完成句子的 expert 权重矩阵。(3) **与 Triton Inference Server 集成**——利用 Triton 的模型管理、动态 batching 和云规模弹性扩缩容实现生产部署。
  - 实验比较：(a) Batch pruning 的 throughput 对比：有/无 batch pruning 优化（1.14× speedup）；(b) 端到端推理吞吐对比：Torch-FP16 vs FT-FP16 vs FT-INT8 vs FT-INT4 在不同 batch size 和 beam 下的每秒处理 tokens 数（Table 3）；(c) 部署成本对比：优化后 NVIDIA T4 GPU 上的 5.32B MoE 模型 vs CPU 上的 0.04B 小模型 vs CPU 上的 5.32B 大模型，比较每月每 token 成本（Table 4）。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100（开发和评估），Docker Ubuntu 20.04 + CUDA 11.6
  - 生产部署：单卡 NVIDIA T4（16GB VRAM），Azure NC4as T4 v3 实例，价格为 $390.55/月
  - CPU baseline：Azure F16s 实例（AVX512），$587.65/月

- 开源Serving框架是什么。修改了什么。
  - 开源框架：NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer），集成 Triton Inference Server（https://github.com/triton-inference-server/server）
  - 修改内容：(1) 在 FasterTransformer 中新增 MoE layer 支持——实现了 MoE encoder layer 和 decoder layer，包含 token routing（基于 CUB radix sort）、expert computation（基于 CUTLASS Grouped GEMM）、以及 TUPE attention；(2) 新增 batch pruning 机制——在 decoder beam search 的每次迭代中，gating 函数检测已完成句子并将其 expert_idx 设为极大值，token routing 将已完成句子排列到激活矩阵末尾，仅对 active_tokens 行执行 expert GEMM；(3) 新增 4-bit/8-bit 量化 expert weights 的 fused GEMM+Dequantize kernel；(4) 支持 Triton Inference Server 的 dynamic batching 集成用于云规模部署。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文基于开源 FasterTransformer 框架扩展，FasterTransformer、CUTLASS、CUB、Triton Inference Server、SentencePiece 均为开源。论文未提供独立开源仓库。
  - 框架输入到硬件执行全过程（以 EN-DE 翻译，batch_size=32, beam=2 为例）：
    1. **输入**：32 句英文句子通过 Triton Inference Server HTTP/gRPC 接口到达，Triton 将请求组 batch 发送给 FasterTransformer backend。
    2. **Tokenizer**：FasterTransformer 调用 SentencePiece tokenizer 将文本转为 token IDs（vocab 128K），形成 input_ids tensor。
    3. **Encoder 执行**：24 层 MoE encoder layers。每层的 self-attention（TUPE）使用标准 FP16 GEMM。每两层的 MoE FFN layer（共 12 个 MoE layers）：(a) Router（top-1 gating）为每个 token 计算 softmax weight 并选择 expert_idx；(b) Token Routing——CUB radix sort 按 expert_idx 排序 tokens，permute activation 使同 expert 的 tokens 连续排列；(c) Expert Computation——CUTLASS Grouped GEMM 并行执行所有 experts 的矩阵乘法，若使用量化则 fused dequantize；(d) Un-permute——恢复原始 token 顺序并乘以 expert scale。输出 encoder hidden states。
    4. **Decoder 执行**（自回归 + beam search）：12 层 decoder layers。每步生成一个 token。在 beam search 中，当某句子的 EOS token 生成后，Batch Pruning 机制将其 expert_idx 设为极大值（如 INT_MAX），token routing 将该句子的 tokens 排列到激活矩阵末尾，subsequent expert GEMM 仅处理前 active_tokens 行——避免为已完成句子加载 expert 权重矩阵。1.14× 加速。
    5. **输出**：生成的 target tokens 通过 Triton Inference Server 流式返回给客户端，de-tokenize 为德文文本。
    6. **部署**：Triton 管理模型实例的生命周期，根据请求流量动态扩缩容（scale up/down），所有实例加载同一 5.32B MoE INT4 量化模型到 T4 GPU（约 1.25GB）。
    7. **成本**：T4 上 5.32B MoE INT4 模型每月每 token 成本 $0.153，低于 CPU 上 0.04B 小模型的 $0.209（且 BLEU 质量更高）。
