## Load-Aware Thresholding in Expert Parallelism (EP 负载感知阈值调整)

术语解释
Load-Aware Thresholding 是在 MoE 分布式推理的 EP 场景下，根据各 device 实际负载动态调整 token-expert computation dropping 阈值的调度策略。Overloaded device 使用更高阈值激进丢弃更多计算，underloaded device 使用更低阈值保守保留更多计算，以最小化 accuracy loss 实现负载均衡和整体加速。

术语是什么？
标准 EP 推理的负载不均问题：因 expert 选择在 batch 内不均衡，不同 device 上分布的 token/expert 计算量差异大，总体推理时间由最繁忙 device 决定。均匀阈值丢弃策略对所有 device 应用相同 drop rate，导致：(a) underloaded device 上不必要的 accuracy loss；(b) overloaded device 上不够激进的 dropping。Load-Aware Thresholding 对每 device 计算 actual_load / ideal_balanced_load 比值：ratio > 1 → 使用预定义 T_max（高阈值强丢弃）；ratio < 1 → 按比例降低阈值 T_eff = T_base × ratio，仅丢弃够平衡负载的最少量计算。

从系统架构角度拆解术语：
```
=== Load-Aware Thresholding in EP ===
N_EP devices, token batch distributed by gating

For each MoE layer:
  # Step 1: Collect per-device load
  load[d] = Σ_{token} count_experts_routed_to_device(d)
  load_ideal = Σ_d load[d] / N_EP
  ratio[d] = load[d] / load_ideal

  # Step 2: Adjust per-device threshold
  For device d in 1..N_EP:
    if ratio[d] > 1:  # overloaded → aggressive dropping
      T_eff_major[d] = T_max_major
      T_eff_minor[d] = T_max_minor
    else:              # underloaded → proportional dropping
      T_eff_major[d] = T_base_major * ratio[d]
      T_eff_minor[d] = T_base_minor * ratio[d]

  # Step 3: Apply 2T-Drop with device-specific thresholds
  For each token-expert pair (t, e_j) on device d:
    s_norm = normalized gating score
    if s_norm < T_eff_major[d]: skip
    elif s_norm < T_eff_minor[d]: compute major sub-expert
    else: compute full expert

  # Step 4: AlltoAll to return results
```
最终效果：overloaded device 通过激进 dropping 将负载降至 balanced 水平，underloaded device 因低阈值几乎不丢弃计算保留更多 accuracy。DeepSeek-V2-Lite-Chat on 8×H20 EP=8: 2T-Drop + load-aware → 1.41× MoE module speedup, 1.13× end-to-end, 仅 0.5% avg accuracy loss。

术语一般如何实现？如何使用？
- 实现：SGLang EP scheduler 中添加 inter-device load 聚合步骤（gather per-device token counts），每 device 本地计算 ratio 并调整 threshold
- 控制开销：仅需在每层 gather 1 个 scalar per device (N_EP-1 communication)，开销可忽略
- Threshold mapping：因 drop rate 与 threshold 非线性（图 12），需 per-model calibration 建立 threshold→drop rate 映射 → 给定 target max drop rate 倒推 T_max
- S-ETP 协同：S-ETP 简化 communication pattern，使 load-aware scheduling 的实现更简洁（无需处理 TP 分片的额外同步）
- 局限：(a) 需要 per-model per-hardware calibration；(b) 对 batch size 变化敏感的推理场景需动态调整 T_max；(c) 与 S-ETP 耦合使用效果最佳

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
