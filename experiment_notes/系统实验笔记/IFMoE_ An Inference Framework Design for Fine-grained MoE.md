## IFMoE: An Inference Framework Design for Fine-grained MoE

- 属于Serving调度的实现是什么？实验比较什么？
  - IFMoE 提出针对 fine-grained MoE 模型的推理框架，包含两个核心实现：
    1. **并行机制重设计（EP+TP Hybrid Parallelism）**：传统 Expert Parallelism（EP）在推理时每台机器复制全部非 expert 参数（Attention、Normalization、Shared Expert），导致内存膨胀，限制了 batch size 和 context length。IFMoE 采用 EP+TP 混合并行：Expert 参数仍用 EP 分布，Shared 参数（Attention、Norm、Shared Expert）使用 Tensor Parallelism（TP）切分，避免非 expert 参数的每机全量复制。通信上，用 double All-Gather 替代传统 All-to-All 操作，因为在单节点内推理场景下 All-Gather 通信开销不高于 All-to-All。
    2. **基于 Self-Draft 的 Speculative Decoding**：观察到 fine-grained MoE 用更少 expert 也能保持较好性能，因此用 MoE 模型自身（激活更少 expert，decode_topk Dk=2）作为 draft model，快速生成 α 个 token，然后用全量 expert（encode_topk Ek=6）重新计算 KV-cache 完成 verification。不同于传统 speculative decoding，IFMoE 接受 draft model 的全部输出 token，仅更新 KV-cache。
  - 实验比较：
    - Baselines：Full model（原始 Qwen2-57B-A14B-Instruct / Deepseek-Lite-Chat 全量 expert 推理）
    - IFMoE vs Full model 在 latency 和 throughput 上的对比
    - 下游性能评估：XSum（摘要）、GSM8K（数学）、TruthfulQA（真实性）、IFEval（指令遵循）
    - 结果：IFMoE 在 benchmark 上取得 >30% 推理速度提升和 >30% 吞吐量提升，下游性能与全量模型接近（lossless 近似）

- 硬件平台是什么，配置是什么。
  - **Qwen2-57B-A14B-Instruct**：4× NVIDIA A6000 GPUs
  - **Deepseek-Lite-Chat**：2× NVIDIA A6000 GPUs
  - 节点内 GPU 间通信（单节点多卡推理场景），无跨节点通信需求
  - 论文未明确说明 CPU、内存、互联类型等具体配置

- 开源Serving框架是什么。修改了什么。
  - IFMoE **未开源**（论文 Checklist 明确标注 "IFMoE is still under develop with future features"）。
  - 论文未明确说明基于哪个开源 Serving 框架构建（如 vLLM、TGI 等），以原型系统实现。
  - 核心修改：
    1. **并行策略切换**：从纯 EP 切换为 EP+TP 混合。Shared 参数（Attention、Norm、Shared Expert）从 EP 的每卡全量复制改为 TP 切分。Expert 参数保持 EP 分布，各机器持有不同 expert。
    2. **通信原语替换**：将传统 MoE 的 All-to-All dispatch/combine 替换为 double All-Gather 操作，适应 EP+TP 混合并行的通信模式。
    3. **Decoding 流程改造**：实现 Algorithm 1 的 draft-decode + KV-cache revision 流程。Draft 阶段用 decode_topk Dk=2 激活少量 expert 快速生成 token，每 α=10 步后执行一次 encode（Ek=6 全量 expert）回填 KV-cache。
    4. **GroupedGEMM Kernel 选择**：由于 PyTorch 与 CUDA 12.5 版本兼容性问题，选用 **Cutlass GroupedGEMM** 实现（而非 cuBLAS GroupedGEMM 或 Triton GroupedGEMM）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：未开源。论文 NeurIPS Checklist 回答为 "No"，理由为 "IFMoE is still under develop with future features"。
  - **使用例子与全过程**（基于论文描述还原）：
    1. **输入**：用户请求以 batch 形式到达，每个请求包含 prompt tokens。
    2. **Prefill 阶段**：所有 machines 接收相同输入 tokens（因 EP+TP 混合模式下共享参数已 TP 切分）。Attention 和 Norm 层通过 TP 在所有 machines 上并行计算。Expert 层：每台 machine 上的 router 计算 token-to-expert 分配，通过 double All-Gather 收集各 machine 所需的 expert 输出。由于单节点内通信带宽充足（NVLink/PCIe），double All-Gather 不会成为瓶颈。
    3. **Decode 阶段（IFMoE Draft）**：Decode 时仅激活 Dk=2 个 expert（而非全量 Ek=6）。每个 decode step：router 选 top-2 experts → GroupedGEMM（Cutlass 实现）并行计算各 expert 输出 → combine → 生成 1 个 token。连续执行 α=10 步，所有 draft token 追加到 buffer。
    4. **KV-cache Revision**：每 α 步后，对 buffer 中所有 token 用 Ek=6 全量 experts 重新做一次 encode forward，更新 KV-cache 中对应位置的 key/value。此步骤确保后续 decode 的 attention 计算基于"全量 expert 应产生的 KV"。
    5. **输出**：生成的 token 序列返回给用户。IFMoE 内存节省使更大 batch size 成为可能（Qwen2 可达 batch size 256，Deepseek-Lite 可达 200）。
