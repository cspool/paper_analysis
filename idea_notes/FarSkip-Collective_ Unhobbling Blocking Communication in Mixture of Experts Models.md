## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- baseline方法是什么？
  - **Baseline**: 标准 MoE Transformer 架构，使用标准的残差连接（residual connections），即每个子块的输出立即通过残差加到主路径上，下一子块的计算必须等待当前子块的完整输出。在分布式执行中，这种"严格顺序依赖"导致以下阻塞通信模式：
    1. **训练中的阻塞 all-to-all**：在 EP 下，MoE 层的 Dispatch（token 发送到对应 expert rank）和 Combine（专家输出聚合回原 rank）均为 all-to-all 集合通信。Dispatch 必须在 attention 完成后、routed expert 计算前执行；Combine 必须在 routed expert 计算后、下一层计算前执行。两者均无法与计算重叠。
    2. **推理中的阻塞 all-reduce**：在 vLLM/SGLang 的 EP 推理实现中，各 rank 计算本地 expert 后通过 all-reduce 聚合结果。Attention 层的 TP 输出投影（RowParallelLinear）也包含 all-reduce。这些 all-reduce 均为同步阻塞调用。
  - 全栈执行例子（以 DeepSeek-V2-Lite 16B 在 Megatron-LM EP=8 训练，单节点 MI325X 8GPU 为例）：
    - **模型推理算法层**：标准 MoE Transformer 层，每层包含 Attention（MLA）+ MoE（64 experts, top-k routing, shared experts）。残差连接：`o_k = o_{k-1} + f_k(o_{k-1})`，即 `f_k` 必须等待 `o_{k-1}` 完全就绪。
    - **系统框架层**：Megatron-LM 训练框架。每层执行顺序：① Attention 子块（含 layer-norm）→ ② 可能的 TP 后 attention all-reduce → ③ layer-norm + gating + router → ④ **Dispatch all-to-all（阻塞）** → ⑤ routed expert 计算 + shared expert 计算 → ⑥ **Combine all-to-all（阻塞）** → 输出到下一层。Dispatch 和 Combine 期间 GPU 计算单元空闲（通信气泡）。
    - **编译框架层**：论文未明确说明。PyTorch + NCCL 通信后端。
    - **kernel调度层**：NCCL all-to-all collective kernel 在执行期间占用 GPU SM（通信计算单元），计算 kernel（expert GEMM）必须等待通信完成。无计算-通信重叠。
    - **硬件架构层**：AMD MI325X 8GPU 单节点，节点内高带宽互联。通信时间占 layer 总时间的显著部分。
  - **Baseline 痛点**：
    1. **阻塞通信导致 GPU 计算资源空闲**（核心痛点）：Dispatch 和 Combine all-to-all 期间，GPU 的 CUDA cores/Tensor cores 无法执行有用的计算，造成"通信气泡"（communication bubble）。随着硬件计算速度提升和 MoE 规模增大（更多 experts、更大 EP 度），通信时间在端到端延迟中的占比不断增大。
    2. **直接加载 FarSkip 架构权重导致性能崩溃**（Fig. 3）：如果不经训练直接将原始 checkpoint 加载到修改后的 FarSkip 连接架构中，模型性能随修改层数增加急剧下降——全部层修改后 MMLU 达到随机基线、HumanEval+ 为 0%。这是因为模型接收到的输入激活值与训练时的分布完全不同（OOD）。
    3. **SFT 微调不足以恢复性能**（Tab. 1-2）：仅用 SFT 数据微调 FarSkip 修改后的模型，在下游任务上显著劣于原始模型（DeepSeek-V2-Lite SFT 平均 55.0 vs 原始 64.5），尤其是在生成任务上（HumanEval+ 仅 11.0 vs 40.2）。SFT 缺乏足够的粒度信号来恢复原始模型的内部表征。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FarSkip-Collective 方法**：通过修改模型架构的连接性（residual connectivity）来消除子块之间的严格顺序依赖，使计算能够在通信进行期间继续执行。两项核心设计：
    1. **FarSkip-Collective 架构修改**（解决痛点 1）：将下一子块的输入从完整的最新激活值改为可用的"过时"或"部分"激活值——
       - 对于 Attention 子块输入（partial）：`attn-in_k = o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}`，省略 `routed-exp-out_{k-1}`。这使得 Combine all-to-all 可与 Attention 计算重叠（因为 Attention 输入不需要 routed expert 输出）。
       - 对于 MLP 子块输入（outdated）：`mlp-in_k = o_{k-1}`。这使得 Dispatch all-to-all 可与 Attention 计算重叠（因为 MLP 输入不需要最新 attention 输出）。
       - 数学上，`o_k = o_0 + f_1(o_0) + f_2(o_1^*) + ... + f_k(o_{k-1}^*)`，每个 `f_i` 仍贡献到残差路径，仅输入激活值不同。所有未来层 `f_j (j ≥ k+2)` 都能访问完整的 `f_k` 输出，保证信息最终不丢失。
    2. **FCSD（FarSkip-Collective Self-Distill）**（解决痛点 2 和 3）：以原始模型为 teacher，FarSkip 修改后的模型为 student，使用 KL 散度知识蒸馏进行训练。关键配方发现：
       - KL 散度 loss（而非 SFT cross-entropy）：teacher 的概率分布提供更细粒度的训练信号，帮助恢复原始模型的内部表征
       - 大 batch-size（2^17-2^18 tokens）和大 learning rate（2e-5−8e-5）：通过 short sweep 确定
       - 全参数训练（不冻结 embedding/LM-head）
       - MBPP+ early stopping：防止训练后期出现 mode collapse
       - 成本：约 100-1000× 低于从头预训练（< 10B tokens vs 数万亿 tokens）

  - 全栈执行例子（与 baseline 同配置，DeepSeek-V2-Lite FarSkip 在 Megatron-LM EP=8 训练）：
    - **模型推理算法层**：FarSkip-Collective 修改后的 MoE 架构，通过 FCSD 蒸馏恢复准确率。以 DeepSeek-V2-Lite 为例，FCSD 模型在 11 个 benchmark 上平均 62.0 vs 原始 64.5（−2.5），SFT baseline 仅 55.0（−9.5）。Llama-4-Scout 109B：FCSD 平均 75.1 vs 原始 76.0（−0.9）。
    - **系统框架层**：修改 Megatron-LM 执行顺序——
      - 前向：① MLA q/k/v 准备 → ② 同步上层的 Combine → ③ gating → ④ **异步 Dispatch**（async_op=True）→ ⑤ core attention + output projection（**与 Dispatch 重叠**）→ ⑥ 同步 Dispatch → ⑦ routed experts → ⑧ **异步 Combine**（async_op=True）→ ⑨ shared experts（**与 Combine 重叠**）
      - 反向：使用两项新技术——① **Stateful Async All-to-All Autograd Function**：在 stateful dictionary 中存储前向和反向通信 handles，通过 backward hook 在输入被访问前同步通信；② **Sequence Number Hijacking**：利用 PyTorch autograd 的 Sequence Number 内部机制，重新排序反向节点优先级——提高子块计算节点优先级，降低通向通信输入的节点优先级，使计算在通信等待期间优先执行。
    - **编译框架层**：论文未明确说明。所有修改在 PyTorch API 层面完成，不涉及编译器修改。
    - **kernel调度层**：基于 PyTorch 的 CUDA Stream 机制和 torch.dist async_op 实现通信-计算重叠。训练中使用两个 CUDA queue：计算 queue（expert GEMM, attention kernel）和通信 queue（NCCL all-to-all）。通过 async_op=True 启动通信后立即返回 handle，计算 queue 继续执行，仅在需要通信结果时调用 handle.wait() 同步。反向通过 backward hook 在 autograd 图中注入同步点。**设计原则：避免 low-level kernel/Triton 修改，保持在 PyTorch API 层面，确保硬件无关性和广泛适用性**。
    - **硬件架构层**：与 baseline 相同（MI325X 8GPU）。单节点训练结果（Tab. 3）：
      - DeepSeek-V2 Lite: 前向重叠率 87.6%, 反向重叠率 89.0%, 总重叠率 88.4%, 端到端加速 1.11×
      - DeepSeek-V3 (L=6): 前向重叠率 92.9%, 反向重叠率 84.1%, 总重叠率 88.9%, 端到端加速 1.04×
      - 多节点强扩展（Fig. 5）：EP=32 时端到端训练加速达 1.22×
    - **推理侧**：
      - Llama-4-Scout (109B) vLLM: all-reduce 重叠率 95.3%, TTFT 加速 12.2%-18.5%
      - DeepSeek-V2 (235B) vLLM: all-reduce 重叠率 97.6%, TTFT 加速 8.2%-16.8%
      - DeepSeek-V3 (671B) SGLang: TTFT 加速 up to 1.34×（TP=8, EP=8）
      - 多节点 decode (TP=16, EP=16, BS=1024): 稳定且一致的 TBT 加速（Fig. 7）

    **核心设计洞察**：FarSkip-Collective 的独特之处在于它是一种"算法-系统协同设计"——不是简单地优化系统实现来隐藏通信（bit-exact 方法），而是在模型架构层面主动消除导致阻塞通信的依赖关系，然后用轻量级知识蒸馏恢复准确率，最后在框架层面实现显式的通信-计算重叠。这种方法比纯系统优化（如 operator decomposition）能覆盖更多的重叠窗口（仅 routed experts + gating 不可重叠），且不依赖特定的硬件特性或 low-level kernel 修改。
