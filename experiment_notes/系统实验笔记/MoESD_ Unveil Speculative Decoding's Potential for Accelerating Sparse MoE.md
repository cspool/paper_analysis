## MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：利用 vLLM 框架的 batched speculative decoding 功能，分析并验证不同 batch size 下 MoE 模型的 SD 加速效果。论文聚焦于（1）中等 batch size（tens of requests）下 SD 对 MoE 的加速潜力——此时所有 expert 已在单步解码中激活，验证多 draft token 不产生额外参数加载开销；（2）Serving 场景分析：private serving（企业内部 chatbot 等中等 batch 场景）、latency-critical 场景（大 batch 不可行）、memory-constrained 场景（MoE 超出 GPU 显存需 offloading）；（3）batch size 对 target efficiency 和 SD speedup 的定量影响趋势。
  - 实验比较：（1）不同 batch size（1-128+）下 SD speedup 曲线——验证先升后降趋势；（2）不同 GPU 平台（2xGPU-A/B, 4xGPU-A/C）的 speedup 对比；（3）MoE vs dense 模型的 end-to-end speedup 随 batch size 变化趋势；（4）不同 sparsity ρ 下 speedup 峰值对应的 batch size 和有效加速范围；（5）不同 dataset/temperature/γ 组合下的 speedup 趋势。

- 硬件平台是什么，配置是什么。
  - 2xGPU-A, 2xGPU-B, 4xGPU-A, 4xGPU-C（论文对 GPU 型号做了匿名化处理）。多 GPU 配置用于评估 inter-GPU parallelization 对 SD speedup 的影响（target model 受益于并行化而 draft model 仍为单 GPU）。

- 开源Serving框架是什么。修改了什么。
  - 框架：vLLM（支持 batched speculative decoding、cudagraph optimization，可报告 T_D, T_T, T_reject, σ 等详细数据）。
  - 修改内容：论文未明确说明对 vLLM 的代码级修改。主要通过修改模型 config.json 中的 `num_experts_per_token` 参数来控制 MoE sparsity 进行实验。vLLM 原生的 batched SD 能力被直接用于验证理论预测。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源代码仓库，实验完全基于开源 vLLM 框架。
  - 基于开源文档和论文，vLLM Serving 全流程：
    1. **输入**：B 个 user requests（prompt tokens）进入 vLLM server。
    2. **Prefill**：vLLM 对 B 个 prompts 做并行 prefill，计算 KV cache。
    3. **Speculative Decode Loop**：每轮执行 Draft → Verify → Reject 三步：
       a. Draft 阶段：draft model（Qwen2-0.5B 或 Eagle head）在单 GPU 上自回归生成 γ 个 draft tokens（每 token 耗时 T_D(B,1)），γ 次 forward 共耗时 γ × T_D(B,1)。
       b. Verify 阶段：target MoE model 以 batch=B, seq_len=γ 做一次并行 forward。MoE layer 内部：Gate 路由 (B×γ) 个 tokens → 每个 token 激活 K 个 expert → N(Bγ) 个不同 expert 被激活并加载参数 → expert FFN 计算 → 加权汇总。vLLM 的 cudagraph 优化捕获并重放计算图。
       c. Reject 阶段：对比 target logits 与 draft logits 做 rejection sampling，丢弃不匹配的后续 token。
    4. **输出**：每个 request 的生成 token 序列返回给用户。
    5. vLLM 报告各阶段时间分解（T_D, T_T, T_reject）和 σ（接受率相关），论文利用这些数据计算 target efficiency 并验证性能模型。
