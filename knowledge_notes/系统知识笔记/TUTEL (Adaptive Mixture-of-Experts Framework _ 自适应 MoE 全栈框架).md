## TUTEL (Adaptive Mixture-of-Experts Framework / 自适应 MoE 全栈框架)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TUTEL 是微软研究院开发的面向大规模 MoE 模型训练和推理的全栈自适应系统（MLSys 2023）。核心洞察：MoE 的 token 路由机制导致每个训练 step 的专家负载（expert capacity）动态变化（实测 4.38× 波动），而传统框架（Fairseq/DeepSpeed）使用静态并行策略和静态流水线度，无法适应这种动态性。TUTEL 通过三大自适应机制解决：(1) 自适应并行切换——基于统一 ZeRO-DP-3 风格张量布局实现 DP ↔ EP+DP+MP 的零成本运行时切换；(2) 自适应流水线——基于 token capacity 分区的多流 CUDA 调度，动态选择流水线度和 All-to-All 算法；(3) 预构建字典——通过 profiling 建立 capacity → 最优策略的映射，运行时 O(1) 查表。开源地址：https://github.com/microsoft/tutel，已集成到 Fairseq 和 DeepSpeed。

从系统架构角度拆解：

TUTEL 在 MoE 训练/推理中的全栈架构和执行流程：

```
[Application Layer — DL Framework]
  PyTorch / Fairseq / DeepSpeed
  ↓ 调用 TUTEL MoE Layer API

[System Layer — TUTEL MoE Framework]

  ┌─ Gating Module ─────────────────────────────────────┐
  │ Dynamic Top-ANY Routing: 每步可调 k 值               │
  │ Dynamic Capacity Factor: capacity_setting={+固定/0自适应/-上界} │
  │ 可选 Cosine Router: 归一化门控 (Eq. 2)                │
  └─────────────────────────────────────────────────────┘
  ↓ gate_output (idxs, scores, locations)

  ┌─ Adaptive Parallelism Selector ───────────────────────┐
  │ 输入: current capacity c = k·f·T/E                     │
  │ 查表: dictionary[⌊c/128⌋] → {r*, d*, a*}             │
  │ r*: 并行策略 (0=DP, 1=EP+DP, 2..max=EP+DP+MP)        │
  │ d*: 流水线度 (1/2/4/8)                                │
  │ a*: All-to-All 算法 (Linear/2DH)                      │
  └─────────────────────────────────────────────────────┘
  ↓ (r*, d*, a*)

  ┌─ Execution Engine ─────────────────────────────────────┐
  │ [Fast Encode]  K0→K1 sparse kernel dispatch input      │
  │ [Flexible A2A] layout (E_g, C, D) dispatch/combine     │
  │ [Multi-Stream Pipeline] d 条 CUDA stream 异步执行:     │
  │   Stream_i: A2A_dispatch(C_i) → Expert_FFN(C_i)        │
  │            → A2A_combine(C_i)                          │
  │            (各stream与相邻stream overlap通信与计算)     │
  │ [Fast Decode]  K1→K2 sparse kernel combine output      │
  └─────────────────────────────────────────────────────┘
  ↓

  ┌─ Communication Backend ───────────────────────────────┐
  │ [Linear A2A]: NCCL P2P, 适合小规模/大消息              │
  │ [2DH A2A]: NCCL P2P 或 MSCCL 编译, 适合大规模/小消息   │
  │           4-phase: stride_memcpy×2 + intra/inter A2A   │
  │ [Flexible A2A]: layout 变换 inline                     │
  └─────────────────────────────────────────────────────┘
  ↓

[Hardware Layer]
  NVIDIA A100 GPU × W (up to 2,048)
  Intra-node: NVLink 3.0 + NVSwitch
  Inter-node: HDR InfiniBand (200 Gbps × 8)
  NCCL 2.10.3-1 + RDMA SHARP plugin
```

从请求/token 在 TUTEL 中的完整生命周期：
1. **Token 到达 MoE 层**：shape (T, D) 的隐藏特征
2. **Dynamic Gating**：Softmax → TopK → capacity_factor 自适应计算（决定实际专家容量 C = k·f·T/E）
3. **Adaptive Selector**：以 ⌊C/128⌋ 查字典，获取本 iteration 的最优 parallelism r、pipelining degree d、A2A algorithm a
4. **Fast Encode (K0→K1)**：SIMT-efficient 稀疏 kernel 生成 dispatch_input (E, C_g, D)
5. **Flexible All-to-All Dispatch**：token 跨 GPU 交换，输出 layout (E_g, C, D)
6. **Expert FFN (Multi-Stream Pipelined)**：沿 C 维度 d-分区，多 CUDA stream 异步执行，重叠 A2A 通信与专家计算
7. **Flexible All-to-All Combine**：专家输出跨 GPU 回传
8. **Fast Decode (K1→K2)**：从 combine 输出恢复 MoE 层输出 (T, D)
9. **输出**：token hidden states，进入下一 Transformer 层

术语一般如何实现？如何使用？

TUTEL 以 pip install 方式安装（`pip install tutel`），通过替换 Fairseq/DeepSpeed 的 MoE 层实现来集成。核心 API：`tutel.moe_layer(gate, expert_fn, capacity_setting=0, adaptive_r='auto')`。预构建字典通过一次 profiling pass 完成（所有可能的 parallelism × pipelining × A2A 组合），profile 结果缓存到磁盘。运行时 MoE 层自动使用查表得到的最优配置。已在 SwinV2-MoE（视觉模型）和多种 MoE 配置下验证，最大扩展到 2,048 A100 GPUs。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale
