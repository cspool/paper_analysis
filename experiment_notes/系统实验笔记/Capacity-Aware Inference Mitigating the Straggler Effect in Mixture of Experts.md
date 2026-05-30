## Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  在Megatron-LM分布式推理框架中修改MoE层的token路由和调度逻辑，通过在All-to-All通信前施加expert容量约束（Token Drop）和扩展本地候选集（Expanded Drop）来缓解expert parallelism下的"Straggler Effect"——即高负载expert完成计算慢导致低负载expert和GPU空闲等待同步barrier的问题。
  实验比较：(a) 不同capacity factor γ下各模型单MoE层的加速比（Figure 4）及端到端加速比（Figure 5）；(b) 不同expert-per-GPU配置对加速效果的影响——Mixtral 1-2E/GPU时效果最优（1.85-1.87×），而OLMoE 8E/GPU时加速减弱；(c) 推理延迟分解分析（Figure 6）：expert computation、permutation、communication各阶段时间随γ变化；(d) Device-Level vs Expert-Level容量约束的端到端speedup对比（Qwen3-MoE：1.31× vs 1.23×, γ=1.0/1.5）；(e) 不同workload（batch size 1K-8K, prompt length 0.1K-4K）下的speedup（Table 10）。

- 硬件平台是什么，配置是什么。
  8× NVIDIA H20 GPU。分布式推理策略：8-way Data Parallelism (DP) + 8-way Expert Parallelism (EP)，通过Megatron-LM框架编排。输入batch配置为batch size 8K、sequence length 512，模拟高吞吐实时serving场景。

- 开源Serving框架是什么。修改了什么。
  开源Serving框架：**Megatron-LM**（Shoeybi et al., 2019），用于实现expert parallelism + data parallelism的分布式MoE推理。
  修改内容：在MoE层的forward流程中，在Gate/Router计算之后、All-to-All token dispatch之前插入容量感知逻辑：
  1. **Token Drop**：Router计算gating scores → 根据expert capacity C=γN̄和gating scores对每expert做top-cap token选择 → 超载expert的剩余token被丢弃（score不参与后续dispatch和FFN计算）
  2. **Expanded Drop**：Router计算gating scores → 在top-k基础上扩展候选集为top-k+m（m=本地设备expert数）→ 逐expert应用capacity constraint → 保留的token经All-to-All dispatch到持有对应expert的GPU → expert FFN计算 → All-to-All combine
  3. **Device-Level变体**：将约束粒度从per-expert放宽到per-device aggregat

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE。修改基于Megatron-LM的MoE分布式推理流程。

  **全链路执行过程（以Mixtral-8×7B-Instruct, 8×H20 GPU, 8-way EP + 8-way DP, γ=1.5, batch 8K × seq 512 为例）**：

  1. **输入阶段**：8K个token组成的batch（batch_size=8K, seq_len=512）→ 均匀分配到8个DP group → 每个GPU持有1000 token × 512 seq 的输入tensor [1000, 512, d_model]

  2. **Self-Attention**（各GPU独立）：标准Multi-Head Attention → 输出hidden states [1000, 512, d_model]。Attention层非MoE，所有GPU均持有完整参数副本（DP复制）。

  3. **MoE Gate/Router**（各GPU独立）：hidden states → Gate Linear: W_g ∈ R^{d×8} → softmax → TopK(k=2, dim=-1) → gating_scores [1000×512, 8], topk_indices [1000×512, 2]

  4. **Capacity-Aware Token Drop/Expanded Drop**（本地GPU执行，论文插入点）：
     - Token Drop: scores × topk_mask → 每expert取top-cap=γ×(N×k)/E个token（dim=0 topk）→ 超载expert的低分token被mask
     - Expanded Drop: topk_idx ∪ local_expert_ids → 扩展候选掩码 → 逐expert top-cap筛选
     - 输出final_map [N, 8]标注每个token最终路由到哪些expert

  5. **All-to-All Token Dispatch**（跨GPU通信）：根据final_map中expert→GPU映射，通过NCCL All-to-All将token发送到持有对应expert的GPU。因capacity constraint减少了超载expert的token数，此次通信数据量减小。

  6. **Expert FFN Computation**（各GPU本地）：收到token的各GPU执行expert FFN：x → W_gate [d, d_ff] → GeLU → W_up [d_ff, d_ff] → × W_down [d_ff, d] → 输出。Mixtral每GPU 1-2个expert，各expert处理的token数因capacity constraint更均衡→ 减少GPU空闲等待。

  7. **All-to-All Combine**（跨GPU通信）：FFN输出通过All-to-All返回原token所在GPU。

  8. **输出Merge**：gate score加权求和各expert输出 → residual add → 输出至下一Transformer层。

  9. **端到端**：重复步骤2-8共32层（Mixtral-8×7B有32层，MoE替换alternate layers的FFN）→ final LM head → token预测。

  **关键性能影响（Figure 6 latency breakdown）**：
  - 无capacity constraint时：expert computation + permutation + communication占主导，gate processing耗时可忽略
  - Token Drop/Expanded Drop后：expert computation显著减少（因丢弃超载expert token），permutation和communication时间也相应减少
  - Expanded Drop扩展到跨设备global experts时communication增加（需传输扩展token）→ 论文因此限制扩展仅在本地设备内

  **加速比受expert-per-GPU配置影响的核心原因**：EP下每GPU托管n_l个expert，总load为n_l个expert的token数之和。若n_l大（如OLMoE 8E/GPU），单个straggler expert load占总load比例小，capacity constraint减少的load比例也小→加速效果削弱。若n_l小（如Mixtral 1-2E/GPU），straggler expert load占比大→容量约束效果显著→1.85-1.87×加速。因此论文建议分配更多GPU做expert分布以增强效果。
