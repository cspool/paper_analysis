## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- 属于Serving调度的实现是什么？实验比较什么？
  实现包括两个层面的 Serving 调度创新：(1) **Load-Aware Thresholding in Expert Parallelism**：在 MoE 分布式推理中，EP 设备间的负载不均是限制效率的主要因素。本文提出基于设备负载动态调整 token-expert 计算丢弃率的机制，采用 step-down thresholding 策略——高负载设备使用更高的丢弃阈值（激进丢弃更多 token-expert 计算），低负载设备使用更低的丢弃阈值（保守保留更多计算）。具体实现：计算每个设备的 actual load / ideal balanced load 比值，若比值 > 1 则阈值设为预定义的最大值，若比值 < 1 则按比例降低阈值，确保所有设备以最小精度损失实现负载均衡。(2) **Soft Expert-Tensor Parallelism (S-ETP)**：从算法层面实现 TP-like 效果而非法仅系统层面修改框架。S-ETP 通过 partial transformation（expert partition + EP）替代传统的 Expert-Tensor Parallelism (ETP)，仅需 AlltoAll 通信（vs ETP 的 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll"），减少 kernel launch 和同步开销，提高 interconnect link utilization。实验比较：(a) ETP vs S-ETP 在不同 EP/TP 配置下的通信带宽（real-world 8×H20 测试 + ASTRA-SIM 模拟 NVL72 和 CloudMatrix384）；(b) load-aware thresholding 下的 speedup vs accuracy trade-off（1T-Drop、2T-Drop、2T-Drop+load-aware 三者对比）。

- 硬件平台是什么，配置是什么。
  8×NVIDIA H20 GPU 服务器节点（单机 8 H20），使用 PyTorch Distributed framework + NCCL backend。模拟环境：NVL72 (NVIDIA GB200, EP=9, TP=8)、CloudMatrix384 (CM384, EP=48, TP=8)，使用 ASTRA-SIM 模拟器进行大规模通信仿真。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**SGLang**（https://github.com/sgl-project/sglang），支持 Mixtral、OLMoE、DeepSeek 等 MoE 模型的高效分布式推理。论文修改/新增：(a) 在 SGLang 框架中实现 DualSparse-MoE inference system，包括 token-expert computation dropping（1T-Drop/2T-Drop）、load-aware thresholding 和 expert partition processing；(b) 实现 S-ETP 通信模式，将 ETP 的 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll" 通信模式简化为单 AlltoAll；(c) 实现 expert partition 的 preprocessing 和 inference 阶段的 neuron reconstruction 及 dual-threshold 控制逻辑；(d) 实现基于设备负载实时调整 drop threshold 的 load-aware 调度策略。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立的开源代码仓库，但基于开源 SGLang 框架实现。DualSparse-MoE 的 Load-Aware EP 推理全流程如下：

  ```
  Input: batch of tokens, model partitioned across N EP devices

  === Preprocessing (one-time, static) ===
  For each expert on each device:
    Importance profiling on calibration samples (MMLU)
    Sort neurons by importance, split into major/minor sub-experts
    Store the reconstructed expert weights

  === Inference Per-Layer Flow (with Load-Aware 2T-Drop) ===

  Step 1: Gating (each device)
    Input token hidden states x on device d
    Compute gating logits: l = x · W_g
    Top-K selection: {e_1, ..., e_K} = TopK(l, K)
    Normalize gating scores of selected experts

  Step 2: Load-Aware Threshold Adjustment (each device)
    // Communication: gather load info across EP devices
    load_d = count of tokens routed to experts on device d
    load_ideal = total_tokens / N_EP
    ratio_d = load_d / load_ideal

    For each token-expert pair on device d:
      if ratio_d > 1:  // overloaded device
        T_eff = T_max (预定义固定值)
      else:
        T_eff = T_base * ratio_d  // proportionally reduced

  Step 3: Dual-Threshold Token-Expert Dropping (each device)
    For each token-expert pair (t, e_j):
      s_norm = normalized gating score
      if s_norm < T_eff_major: skip computation  // T_major from 2T-Drop
      elif s_norm < T_eff_minor: compute only major sub-expert
      else: compute full expert (major + minor)

  Step 4: Expert Computation + AlltoAll
    Compute selected (sub-)expert FFN outputs locally
    AlltoAll: exchange token results back to original devices
    (S-ETP variant: single AlltoAll vs ETP's multi-stage communication)

  Step 5: Combine Results
    y = weighted sum of computed expert outputs by original gating scores
    Continue to next layer

  === Communication Patterns: ETP vs S-ETP ===
  ETP (Expert-Tensor Parallelism):
    AlltoAll → TP AllGather | ReduceScatter → AlltoAll
    Multiple kernel launches + synchronization barriers

  S-ETP (Soft Expert-Tensor Parallelism, 本文):
    AlltoAll (single operation)
    Expert partition via partial transformation handles TP splitting algorithmically
    Result: 3.0%-29.9% bandwidth improvement on real H20; 10.2%-80.4% on NVL72 simulation
  ```

  **S-ETP bandwidth improvement 实测结果**：EP=4, TP=2 on 8×H20：3.0%-29.9%；EP=2, TP=4 on 8×H20：9.2%-15.2%。模拟环境：NVL72 (EP=9, TP=8)：10.2%-80.4%；CloudMatrix384 (EP=48, TP=8)：9.9%-28.3%。

  **Load-Aware 最终结果**：DeepSeek-V2-Lite-Chat on 8×H20 with EP=8：2T-Drop + load-aware thresholding → 1.41× MoE module speedup、1.13× end-to-end speedup，仅 0.5% average accuracy loss。
