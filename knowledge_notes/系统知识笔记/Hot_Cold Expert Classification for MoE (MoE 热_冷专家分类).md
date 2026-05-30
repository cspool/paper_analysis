## Hot/Cold Expert Classification for MoE (MoE 热/冷专家分类)

术语解释
Hot/Cold Expert Classification 是根据 expert 在推理时的激活频率或重要性将其分为"热"（频繁使用）和"冷"（偶尔使用）两类的策略，用于指导 GPU-NDP 系统中的 expert 放置——hot experts 在 GPU HBM 全精度执行，cold experts 在 NDP 设备量化执行。

术语是什么？
Hot/Cold 分类源于 MoE 推理中的一个关键观察：expert 激活高度倾斜——少数 experts 处理大部分 token，多数 experts 很少被使用。Mixtral-8×7B 在 WikiText-2 上的 expert 激活分布展示了这种长尾模式。

分类策略的演进：
1. **Global Frequency (MoNDE)**：基于全局历史统计，激活次数多的 → hot, 少的 → cold。缺点：忽略 context dependence（同一 expert 对不同输入热度不同）
2. **Per-Sequence Prefill Statistics (本论文)**：基于当前序列 prefill 阶段的 (P_{l,e}, W_{l,e}) → 重要性分数 S_{l,e} → top-K hot, rest cold。优点：捕捉 context-dependent 激活模式，prefill-decode 相似度 0.89 保证可靠性
3. **Hybrid**：全局 + per-sequence → 全局先验初始化为 hot set，per-sequence 微调

从系统架构角度拆解术语：
Hot/Cold 分类在 GPU-NDP 调度中的决策树：

```
每层 l 的 expert e:
  ┌─ S_{l,e} 计算 ─┐
  │  P_{l,e}: 激活次数         │
  │  W_{l,e}: 路由评分总和      │
  │  S_{l,e} = 0.5P̃ + 0.5W̃    │
  └─────────────────┘
           ↓
  ┌─ 排序 ──────────┐
  │ argsort(S, desc)          │
  └─────────────────┘
           ↓
  ┌─ Hot/Cold 分类 ────────┐
  │ Top-K → HOT (GPU, FP16)       │
  │  - 高激活频率 + 高路由评分    │
  │  - 全精度保证关键 expert 质量  │
  │  - K 由 GPU HBM 容量限制      │
  │                              │
  │ Rest → COLD (NDP, 1-4 bit)    │
  │  - 低激活频率 或 低路由评分   │
  │  - 量化降低 NDP 计算压力      │
  │  - bitwidth 按重要性分配      │
  └──────────────────────────────┘
```

GPU/NDP expert 比例示例：
- Mixtral-8×7B (8 experts/layer, 80GB HBM): K=4 → 4 hot (GPU FP16) + 4 cold (NDP 1-4 bit)
- Mixtral-8×22B (8 experts/layer): K=2 → 2 hot (GPU FP16) + 6 cold (NDP 1-4 bit)

术语一般如何实现？如何使用？
- 分类粒度：per-layer（不同层的 expert 激活模式不同）
- 分类频率：per-sequence（once after prefill）
- K 选择：GPU 可用 HBM / expert FP16 size → 如 Mixtral-8×7B 每 expert ~5.6GB FP16, 80GB HBM 除 attention+KV cache 后约可容纳 4 experts
- 适用场景：GPU-NDP 异构系统、expert offloading、混合精度推理
- 关键权衡：K 越大 → 精度越接近 FP16 baseline，但 NDP 执行 expert 越少 → NDP 利用率越低

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
