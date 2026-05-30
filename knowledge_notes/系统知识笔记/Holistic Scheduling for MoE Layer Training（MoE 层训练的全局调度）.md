## Holistic Scheduling for MoE Layer Training（MoE 层训练的全局调度）

术语是什么？
Holistic Scheduling for MoE Layer Training 是 MegaScale-MoE 提出的一种系统级调度策略，将每个 MoE Transformer 层的 attention 和 FFN 模块分解为独立的 GPU kernel 算子（而非依赖 torch.autograd 的 monolithic 自动微分），在一个统一的 macro module 中手动编排前向和反向传播的所有计算与通信算子的执行顺序。相比于 Megatron-LM 依赖 PyTorch 自动微分系统（torch.autograd）进行反向传播，Holistic Scheduling 可以实现：
(1) 通信算子与无依赖的计算算子在独立的 CUDA stream 上异步并发执行（inter-operator overlap）；
(2) 反向传播中将 activation recomputation 与 gradient communication 交织调度，使重计算延迟被通信完全掩盖；
(3) Selective activation rematerialization：仅保留计算密集的中间激活（如 GroupedGEMM 的输出），丢弃可由通信或轻量计算重新获得的激活（如 fc2_in 通过 recompute SiLU + fc3_out 获得），节省约 50% 的激活内存，且不增加训练时间。

从系统架构角度拆解术语：
MegaScale-MoE 的 Holistic Scheduling 以一个 MoE 层为例的完整流程（基于论文 §4.1 和 Figure 8）：

**前向传播算子链**：
1. hidden [b, s/n, h] → RMSNorm → ln1_out
2. QKV Projection (GEMM+A2A fused kernel) → qkv [b, s, h(1+2/m)/n]
3. SelfAttention (FlashAttention) → attn [b, s, h/n]
4. Output Projection (A2A+GEMM fused kernel) → attn_out [b, s/n, h]
5. Residual Add → ln2_in → RMSNorm → ln2_out
6. Expert dispatch: All-Gather → Scatter → ffn_in [b*s*k/n, h]
7. SwiGLU FFN (fused kernel): fc1_out, fc3_out → SiLU gating → fc2_out
8. Expert combine: Gather → Reduce-Scatter → ffn_out [b, s/n, h]
9. Residual Add → hidden(next)

**反向传播调度（含 Selective Activation Rematerialization）**：
- 仅保留 highlighted 激活（论文 Figure 9）：hidden, ln1_out, qkv, q_rope, k_rope, qkv_a2a, attn, ln2_out, ln2_out_ag, fc2_in, fc2_out
- 丢弃的激活在 backward 时重计算并与通信重叠：
  - fc2_in：通过 re-performing SiLU(fc1_out, fc3_out) 重新获得，与 gradient all-gather (Δffn_out) 并发
  - ffn_in：通过 re-performing RMSNorm(ln2_in) + all-gather 重新获得，隐藏在 FC2 GroupedGEMM 内
- 将 ffn_out 的加权求和提前到 SwiGLU 激活函数后立即执行（避免跨非线性边界），消除存储 ffn_out 的需求
- 激活内存从 (2n + 2k + 3kf + 12 + 5/m)bsh/n 降至 (2kf + 4 + 2/m)bsh/n，约 50% 降低

**资源协调**：
- 统一的 macro module 从 caller 视角管理整个 layer，扩大调度灵活性
- Runtime 协调并发通信任务：决定每个通信算子的 SM 分配数量，避免资源冲突和阻塞
- 多 CUDA stream 异步执行，通过 CUDA event 同步

术语一般如何实现？
- 基于 Megatron-LM（开源 github.com/NVIDIA/Megatron-LM）构建，将其 MoE layer 中依赖 torch.autograd 的 monolithic 执行替换为手动分解的算子序列
- 使用多个 CUDA stream：主计算 stream + 通信 stream(s)，通过 cudaEventRecord/cudaStreamWaitEvent 同步
- Selective activation rematerialization 的保留/丢弃决策原则：保留 GEMM 类算子（计算密集）输出，丢弃 RMSNorm/通信类算子（memory-bound）输出
- 论文的 Holistic Scheduling 是目前手动 hand-tailored 的，作者明确指出未来工作方向是自动化搜索最优调度（Holistic vs. Automatic, §7）
- 实际部署于 ByteDance 生产环境训练千亿参数 MoE 模型，单训练任务可扩展至 10,000+ GPU 持续数月
- 论文未公开源代码（为 ByteDance 内部生产系统）

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
