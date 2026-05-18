## Headroom-maximizing LLM Scheduler（最大化空闲利用率的LLM调度器）

术语是什么？通过联网搜索让回答具体和精准。

Headroom-maximizing LLM Scheduler 是 LEGO 的系统侧核心组件，负责在游戏-LLM 共置场景中动态预测 GPU rendering headroom、选择跳层策略、并将 LLM 推理拆分为细粒度 subtask 以最大化利用碎片化 GPU 空闲时间，同时保证游戏渲染和 LLM 推理的 SLO。调度器由三部分组成：(1) LR-based headroom predictor：预测下一个 execution window 内的总渲染 headroom；(2) Layer-skipping strategy selector：根据预测 headroom 选择跳层数和对应 adaptor；(3) Feedback-driven subtask dispatcher：基于 rendering subtask 状态实时调度 fine-grained/coarse-grained LLM subtasks。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

调度器运转流程（以 BlackMyth + Llama3-8B + 200 APM 为例）：

```
Step 1: Headroom Prediction
  - 时间单位: execution window = 300ms (= 60000/200 APM)
  - 一个 window 包含 18 个 rendering frames (= 300/16.6)
  - LR 模型输入: 前三个 execution windows 的总 rendering headroom
  - 预测: 下一个 window 的总 rendering headroom
  - 预测精度: 最大误差 1.3%（BlackMyth 300APM），平均误差 0.6%
  - 原因: window 跨 12-36 帧，单帧波动被平滑

Step 2: Strategy Selection
  - 根据预测 headroom 计算可用的 GPU 时间
  - 扣除 prefill/decode 的固定开销
  - 确定需要跳过的 Transformer layer 数量
  - 选择对应的 adaptor

Step 3: LLM Task Splitting
  - Prefill phase: 按 self-attention (~0.5ms) 和 FFN sublayer (~1.0ms) 划分 subtask
  - Decode phase: 按单个 Transformer layer (~0.4ms) 划分 subtask

Step 4: Runtime Dispatch (Feedback-driven)
  - 监控 rendering subtask start/completion
  - 当 rendering subtask 完成:
      if 下一 rendering subtask 未开始:
          提交 fine-grained LLM subtask (利用 intra-rendering headroom)
      else:
          等待 (避免抢占渲染)
  - 当整帧渲染完成:
      计算 inter-rendering headroom
      提交 coarse-grained LLM subtask (多个 layers)
  - Safety check: 每个 dispatched LLM subtask 满足 T_subtask ≤ T_minimal

Step 5: Spike Handling
  - 每个 token 生成后，用最新 workload data 更新 headroom prediction
  - 检测 QoS violation risk → 动态调整后续 token 的 layer-skipping 策略
```

**为什么不用 time-series 模型做逐帧预测？**
- ARIMA/SVM/LR 在 per-rendering-task 粒度预测时误差 >3%（最大 5.49%）
- ARIMA 预测耗时 ~1s，SVM >50s → 实时不可用
- LEGO 改用 execution window 为单位 + LR → 误差 <1.3%，推理 1.3ms

**为什么 LR 而非更复杂模型？**
- 使用 execution window 作为时间单位后，headroom 序列展现强线性可预测性
- LR 推理开销仅 1.3ms（3-input），运行时 fit 0.9ms
- 与游戏 SLO 时间尺度匹配（100-300ms execution window vs 1-50s predictor overhead）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **框架修改**：修改 llama.cpp 的 traversal function（computation graph creation 与 traversal 分离 → 在 traversal 中加入调度逻辑）
- **集成方式**：llama.cpp front-end 集成到 Unreal Engine 4，通过 dynamic library 调用其他 LLM 功能
- **状态监控**：game engine 维护 rendering task 状态变量，scheduler 通过 polling 获知 rendering subtask start/end
- **新的 schedulable traversal function**：注册到 dynamic library，确保 LLM inference 在任意 subtask boundary 暂停/恢复时保持正确
- **LR 模型**：简单线性回归，fit 3 个历史数据点预测 1 个未来值 → fit 0.9ms，predict 1.3ms
- **Spike 处理**：severe spike 定义为连续两帧间 rendering workload 增加 >50%；仅 1.2% 帧出现此类 spike；在 window 级别（12-36 帧）影响可忽略；multi-frame workload increase 被 LR 的 temporal prediction 捕获

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU
