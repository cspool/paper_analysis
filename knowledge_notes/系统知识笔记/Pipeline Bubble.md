## Pipeline Bubble

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pipeline Bubble（流水线气泡）是流水线并行系统中的空闲时间——当流水线的不同阶段处理时间不均衡或存在依赖约束延迟了某些阶段的启动时，部分计算资源处于等待状态而无法被利用。气泡大小直接影响流水线效率：

$$\text{效率} = \frac{\text{总计算时间}}{\text{总计算时间} + \text{总气泡时间}}$$

FOLDMOE 识别出 attention-MoE pipeline 中的两类气泡来源：
1. **阶段不平衡（Stage Imbalance）**：attention computation 和 expert computation 耗时不同，aAaM 调度导致 A2A combine 只能与较短的 expert comp 重叠
2. **微批次不平衡（Micro-batch Imbalance）**：token-uniform 切片下，causal attention 的后序微批次计算量更大

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 中不同类型调度产生的气泡对比：

```
aAaM (大 bubble):
Attn:  [======mb0======][======mb1======][======mb2======][======mb3======]
                                                                          |
MoE:                                                [A2A-d0][Exp0][A2A-c0][A2A-d1][Exp1][A2A-c1]...
                                                      ↑ 大 bubble: A2A-combine 阶段只能与 Exp 重叠

1A1M (小 bubble):
Attn:  [==mb0==][=====mb1=====][======mb2======][=======mb3=======]
                                                                  |
MoE:      [A2A-d0][Exp0][A2A-c0][A2A-d1][Exp1][A2A-c1][A2A-d2]...
              ↑ A2A-combine 可与 attention 重叠，bubble 大幅减小

1A1M + time-uniform (最小 bubble):
Attn:  [====mb0====][====mb1====][====mb2====][====mb3====]  ← 时间均匀
                                                              |
MoE:      [A2A-d0][Exp0][A2A-c0][A2A-d1][Exp1][A2A-c1]...  ← 完美 overlap
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

减少 pipeline bubble 的通用策略：
1. **增加微批次数量**：更多微批次 → bubble 时间占比更小（但 kernel launch overhead 增大）
2. **均衡阶段耗时**：如 FOLDMOE 的时间均匀切片和 1A1M 调度
3. **异步流水线**：如 PipeDream 的 1F1B（1 forward 1 backward）调度
4. FOLDMOE 的可调参数 d (overlap degree) 控制 bubble 大小和 kernel launch overhead 的 trade-off

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

FarSkip-Collective 将 overlapping 概念从系统层面提升到算法-系统协同设计：不依赖 token-level micro-batching 或 operator decomposition，而是修改模型架构连接性来消除计算图层面的阻塞依赖。这种方法不改变模型参数形状，使用 PyTorch API 层面（async_op + CUDA Stream）的显式实现，支持训练（forward + backward）和推理（prefill + decode）。训练侧在 Megatron-LM 中实现，推理侧在 vLLM/SGLang 中实现。关键是重叠窗口的完整性——仅 routed experts 和 gating 的计算不可重叠（Eq. 9），其余所有计算都可与通信重叠。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
